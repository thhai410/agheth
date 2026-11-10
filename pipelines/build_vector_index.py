"""Parquet (data/processed) → chunk → BGE-M3 → Milvus hybrid collection (M1).

Usage (cần Docker: docker compose up -d):

    python -m pipelines.build_vector_index --max-rows 500            # subset để dev
    python -m pipelines.build_vector_index --drop --max-rows 500     # tạo lại collection
    python -m pipelines.build_vector_index                           # full train split

Tách 2 pha khi embed trên GPU server (server không cần Milvus):

    # trên GPU server — chỉ embed rồi lưu vectors:
    python -m pipelines.build_vector_index --out data/embeddings/train.vectors.parquet \
        --batch-size 256 --device cuda
    # máy local — nạp vectors đã embed, không chạy lại model:
    python -m pipelines.build_vector_index --from-vectors <file.parquet> --drop

Lần đầu sẽ tải model BGE-M3 (~2GB) từ HF. Full 112k rows: embed trên GPU ~10-20 phút,
trên CPU nhiều giờ — nên tách pha (--out trên server, --from-vectors ở local).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd  # noqa: E402

from agent.config import get_settings  # noqa: E402
from agent.data.chunk import Chunk, chunk_row  # noqa: E402
from agent.retrieval.vector import BgeM3Embedder, MilvusVectorStore  # noqa: E402


def load_chunks(split: str, max_rows: int, max_chars: int) -> list[Chunk]:
    parquet_path = REPO_ROOT / "data" / "processed" / f"{split}.parquet"
    if not parquet_path.exists():
        raise SystemExit(f"thiếu {parquet_path} — chạy `make data` trước")
    df = pd.read_parquet(parquet_path)
    if max_rows > 0:
        df = df.head(max_rows)
    print(f"read {parquet_path.name}: {len(df)} rows")
    chunks = []
    for row in df.to_dict(orient="records"):
        chunks.extend(chunk_row(row, max_chars=max_chars))
    print(f"chunks: {len(chunks)}")
    return chunks


def embed_to_parquet(chunks: list[Chunk], out_path: Path, args: argparse.Namespace) -> int:
    """Chỉ embed rồi lưu vectors ra parquet — không cần Milvus (GPU server)."""
    embedder = BgeM3Embedder(device=args.device, batch_size=args.batch_size)
    print(f"embed device: {embedder.device}")
    frames = []
    for start in range(0, len(chunks), args.batch_size):
        batch = chunks[start : start + args.batch_size]
        vectors = embedder.encode_documents([c.text for c in batch])
        rows = MilvusVectorStore._rows_from_chunks(batch, vectors)
        frames.append(pd.DataFrame(rows))
        done = min(start + args.batch_size, len(chunks))
        if start % (args.batch_size * 10) == 0 or done == len(chunks):
            print(f"embedded {done}/{len(chunks)}", flush=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_parquet(out_path, index=False)
    print(f"saved {len(chunks)} vectors -> {out_path}")
    return 0


def insert_vectors(vectors_path: Path, args: argparse.Namespace) -> int:
    """Nạp vectors từ parquet do --out sinh ra — không embed lại (Milvus local)."""
    if not vectors_path.exists():
        print(f"thiếu {vectors_path}")
        return 1
    df = pd.read_parquet(vectors_path)
    print(f"read {vectors_path.name}: {len(df)} rows")
    rows = df.to_dict(orient="records")
    settings = get_settings()
    uri = args.uri or settings.milvus_uri
    store = MilvusVectorStore(uri=uri)
    store.create_collection(drop=args.drop)
    inserted = store.index_rows(rows, batch_size=args.batch_size)
    print(f"milvus[{store.collection}] <- {inserted} chunks (uri={uri})")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Parquet -> chunk -> BGE-M3 -> Milvus")
    parser.add_argument("--split", default="train", choices=["train", "val"])
    parser.add_argument("--max-rows", type=int, default=0, help="0 = full split")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-chars", type=int, default=800)
    parser.add_argument("--drop", action="store_true", help="xoá collection cũ trước khi nạp")
    parser.add_argument("--uri", default=None, help="MILVUS_URI (mặc định từ .env)")
    parser.add_argument("--device", default=None, help="cuda|cpu — mặc định auto-detect")
    parser.add_argument(
        "--out", default=None,
        help="chỉ embed rồi lưu vectors ra parquet (không cần Milvus)",
    )
    parser.add_argument(
        "--from-vectors", default=None, metavar="PARQUET",
        help="nạp vectors từ parquet do --out sinh ra (không embed lại)",
    )
    args = parser.parse_args(argv)

    if args.from_vectors:
        return insert_vectors(Path(args.from_vectors), args)

    chunks = load_chunks(args.split, args.max_rows, args.max_chars)

    if args.out:
        return embed_to_parquet(chunks, Path(args.out), args)

    settings = get_settings()
    uri = args.uri or settings.milvus_uri
    embedder = BgeM3Embedder(device=args.device, batch_size=args.batch_size)
    print(f"embed device: {embedder.device}")
    store = MilvusVectorStore(uri=uri, embedder=embedder)
    store.create_collection(drop=args.drop)
    inserted = store.index_chunks(chunks, batch_size=args.batch_size)
    print(f"milvus[{store.collection}] <- {inserted} chunks (uri={uri})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
