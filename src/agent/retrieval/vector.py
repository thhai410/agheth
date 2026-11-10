"""Milvus hybrid search — BGE-M3 dense + server-side BM25 sparse, RRF fusion (M1).

Adapter interface (architecture.md §4): mọi lời gọi Milvus đi qua module này.
Collection: dense (COSINE) + sparse (BM25 function trên cột `text`), rank RRFRanker.
"""
from __future__ import annotations

from dataclasses import dataclass

from agent.data.chunk import Chunk

DENSE_DIM = 1024  # BGE-M3
COLLECTION = "health_chunks"


def _trunc_utf8(text: str, max_bytes: int) -> str:
    """Cắt chuỗi theo BYTE UTF-8 — Milvus đếm max_length của VARCHAR bằng bytes.

    An toàn dưới cả 2 cách đếm (bytes >= chars): cắt theo byte thì số ký tự
    luôn <= max_bytes, nên không bao giờ vượt max_length.
    """
    raw = text.encode("utf-8")
    if len(raw) <= max_bytes:
        return text
    return raw[:max_bytes].decode("utf-8", errors="ignore")


@dataclass
class Hit:
    text: str
    score: float
    source_row_id: str
    chunk_index: int
    question: str = ""


class BgeM3Embedder:
    """Wrap BGE-M3 (FlagEmbedding) — lazy import, auto-detect CUDA (GPU server) hoặc CPU."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        batch_size: int = 8,
        device: str | None = None,
        fp16: bool = True,
    ):
        import torch
        from FlagEmbedding import BGEM3FlagModel  # lazy — kéo torch/transformers

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.batch_size = batch_size
        # fp16 chỉ có ý nghĩa trên GPU — CPU thì để fp32
        self._model = BGEM3FlagModel(model_name, use_fp16=fp16 and device == "cuda", device=device)

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        out = self._model.encode(
            texts,
            batch_size=self.batch_size,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        return self._to_lists(out["dense_vecs"])

    def encode_query(self, text: str) -> list[float]:
        return self.encode_documents([text])[0]

    @staticmethod
    def _to_lists(vectors) -> list[list[float]]:
        if hasattr(vectors, "tolist"):
            vectors = vectors.tolist()
        return [[float(x) for x in vec] for vec in vectors]


class MilvusVectorStore:
    """Milvus collection hybrid (BM25 function phía server — không gửi sparse từ client)."""

    def __init__(
        self,
        uri: str,
        embedder: BgeM3Embedder | None = None,
        collection: str = COLLECTION,
    ):
        from pymilvus import MilvusClient

        self._client = MilvusClient(uri=uri)
        self._embedder = embedder  # None khi chỉ nạp vectors có sẵn (--from-vectors)
        self.collection = collection

    def create_collection(self, drop: bool = False) -> None:
        from pymilvus import DataType, Function, FunctionType

        if self._client.has_collection(self.collection):
            if not drop:
                return
            self._client.drop_collection(self.collection)

        schema = self._client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("chunk_id", DataType.VARCHAR, is_primary=True, max_length=128)
        schema.add_field("text", DataType.VARCHAR, max_length=8192, enable_analyzer=True)
        schema.add_field("question", DataType.VARCHAR, max_length=4096)
        schema.add_field("source_row_id", DataType.VARCHAR, max_length=64)
        schema.add_field("chunk_index", DataType.INT64)
        schema.add_field("language", DataType.VARCHAR, max_length=8)
        schema.add_field("dense", DataType.FLOAT_VECTOR, dim=DENSE_DIM)
        schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)  # pymilvus 3.x
        schema.add_function(
            Function(
                name="text_bm25",
                input_field_names=["text"],
                output_field_names="sparse",
                function_type=FunctionType.BM25,
            )
        )
        index_params = self._client.prepare_index_params()
        index_params.add_index(
            field_name="dense", index_type="HNSW", metric_type="COSINE",
            params={"M": 16, "efConstruction": 200},
        )
        index_params.add_index(
            field_name="sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25",
        )
        self._client.create_collection(self.collection, schema=schema, index_params=index_params)

    def index_chunks(self, chunks: list[Chunk], batch_size: int = 32) -> int:
        """Embed + insert các Chunk (sparse tính phía server). Trả về số chunk đã nạp."""
        assert self._embedder is not None, "index_chunks cần embedder"
        total = 0
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            vectors = self._embedder.encode_documents([c.text for c in batch])
            total += self.index_rows(
                self._rows_from_chunks(batch, vectors), batch_size=batch_size
            )
        return total

    @staticmethod
    def _rows_from_chunks(chunks: list[Chunk], vectors: list[list[float]]) -> list[dict]:
        return [
            {
                "chunk_id": f"{c.source_row_id}#{c.chunk_index}",
                # truncate phòng thủ theo schema (byte) — dataset có câu hỏi rất dài
                "text": _trunc_utf8(c.text, 8192),
                "question": _trunc_utf8(c.question, 4096),
                "source_row_id": c.source_row_id,
                "chunk_index": c.chunk_index,
                "language": c.language,
                "dense": vec,
            }
            for c, vec in zip(chunks, vectors, strict=True)
        ]

    def index_rows(self, rows: list[dict], batch_size: int = 64) -> int:
        """Nạp rows đã có sẵn dense vector (không embed lại) — dùng cho --from-vectors."""
        total = 0
        for start in range(0, len(rows), batch_size):
            batch = []
            for row in rows[start : start + batch_size]:
                row = dict(row)
                if not isinstance(row["dense"], list):  # np.ndarray từ parquet
                    row["dense"] = [float(x) for x in row["dense"]]
                # truncate byte phòng thủ — vectors parquet cũ có thể chưa cắt
                row["text"] = _trunc_utf8(str(row["text"]), 8192)
                row["question"] = _trunc_utf8(str(row["question"]), 4096)
                batch.append(row)
            self._client.insert(collection_name=self.collection, data=batch)
            total += len(batch)
        return total

    def search(self, query: str, top_k: int = 5) -> list[Hit]:
        """Hybrid search: dense (BGE-M3) + sparse (BM25) → RRF fusion."""
        from pymilvus import AnnSearchRequest, RRFRanker

        query_vec = self._embedder.encode_query(query)
        dense_req = AnnSearchRequest(
            data=[query_vec], anns_field="dense",
            param={"metric_type": "COSINE", "params": {}}, limit=top_k,
        )
        sparse_req = AnnSearchRequest(
            data=[query], anns_field="sparse",
            param={"metric_type": "BM25", "params": {}}, limit=top_k,
        )
        results = self._client.hybrid_search(
            collection_name=self.collection,
            reqs=[dense_req, sparse_req],
            ranker=RRFRanker(),
            limit=top_k,
            output_fields=["text", "question", "source_row_id", "chunk_index"],
        )
        return [
            Hit(
                text=hit["entity"]["text"],
                score=float(hit["distance"]),
                source_row_id=hit["entity"]["source_row_id"],
                chunk_index=int(hit["entity"]["chunk_index"]),
                question=hit["entity"].get("question", ""),
            )
            for hit in (results[0] if results else [])
        ]
