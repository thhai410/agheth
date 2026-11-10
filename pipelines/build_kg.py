"""Parquet (data/processed) → LLM triplet extraction → validate → Neo4j (M1).

Usage (cần Docker: docker compose up -d; LLM_API_KEY trong .env —
DeepSeek hoặc Gemini qua LLM_BASE_URL/LLM_MODEL, xem .env.example):

    python -m pipelines.build_kg --max-rows 200     # demo — 200 lượt gọi LLM
    python -m pipelines.build_kg --full             # CẢ 112k rows — tốn kém, cân nhắc!

Mỗi row = 1 lượt gọi LLM (temperature 0) + validate whitelist predicate
(triplet sai bị loại fail-closed, xem agent.retrieval.kg).
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
from agent.llm import LlmClient  # noqa: E402
from agent.retrieval.kg import Neo4jKgStore, Triplet, validate_triplet  # noqa: E402

SYSTEM_PROMPT = (
    "Bạn là chuyên gia trích xuất tri thức y tế. Từ câu hỏi của bệnh nhân và "
    "câu trả lời của bác sĩ, trích các triplet dạng (subject, predicate, object).\n"
    "predicate CHỈ được chọn trong danh sách: has_symptom, treats, indicates, "
    "contraindicated, associated_with.\n"
    "Chỉ trích thông tin thật sự xuất hiện trong văn bản, không suy diễn.\n"
    'Trả về JSON đúng schema: {"triplets": [{"subject": "...", "predicate": "...", '
    '"object": "..."}]} — nếu không có triplet nào thì {"triplets": []}.'
)

USER_TEMPLATE = "Câu hỏi bệnh nhân:\n{question}\n\nCâu trả lời bác sĩ:\n{answer}"


def extract_triplets(client: LlmClient, row: dict, source_row_id: str) -> list[Triplet]:
    payload = USER_TEMPLATE.format(
        question=(row.get("question") or "")[:2000], answer=(row.get("answer") or "")[:4000]
    )
    data = client.complete_json(SYSTEM_PROMPT, payload)
    triplets = []
    for item in (data.get("triplets") or []):
        if not isinstance(item, dict):
            continue
        triplets.append(
            Triplet(
                subject=str(item.get("subject") or ""),
                predicate=str(item.get("predicate") or ""),
                object=str(item.get("object") or ""),
                source_row_id=source_row_id,
            )
        )
    return triplets


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Parquet -> DeepSeek triplet -> Neo4j")
    parser.add_argument("--split", default="train", choices=["train", "val"])
    parser.add_argument("--max-rows", type=int, default=200, help="số row gửi đi extract")
    parser.add_argument("--full", action="store_true", help="toàn bộ split (rất tốn kém!)")
    args = parser.parse_args(argv)

    parquet_path = REPO_ROOT / "data" / "processed" / f"{args.split}.parquet"
    if not parquet_path.exists():
        print(f"thiếu {parquet_path} — chạy `make data` trước")
        return 1

    df = pd.read_parquet(parquet_path)
    if not args.full:
        df = df.head(args.max_rows)
    print(f"read {parquet_path.name}: {len(df)} rows -> extract triplets")

    settings = get_settings()
    llm = LlmClient(
        api_key=settings.api_key, model=settings.model_id, base_url=settings.llm_base_url
    )
    store = Neo4jKgStore(
        uri=settings.neo4j_uri, user=settings.neo4j_user, password=settings.neo4j_password
    )
    store.ensure_schema()

    collected: list[Triplet] = []
    invalid = 0
    for i, row in enumerate(df.to_dict(orient="records")):
        source_row_id = str(row.get("id") or i)
        try:
            triplets = extract_triplets(llm, row, source_row_id)
        except Exception as exc:  # noqa: BLE001 — 1 row lỗi không được chặn cả pipeline
            print(f"  row {source_row_id}: LLM error: {exc}")
            continue
        for triplet in triplets:
            if validate_triplet(triplet):
                invalid += 1
            else:
                collected.append(triplet)
        if (i + 1) % 50 == 0:
            print(f"  ... {i + 1}/{len(df)} rows, {len(collected)} triplets hợp lệ")

    report = store.ingest(collected)
    store.close()
    print(f"done: extract {len(collected) + invalid} triplets, loại {invalid} khi validate")
    print(f"neo4j ingest: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
