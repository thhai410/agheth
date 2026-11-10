"""HF → clean → Parquet + stats.json (architecture.md §5, milestone M0).

Usage (chạy từ repo root):

    python -m pipelines.load_dataset --max-rows 200   # sample nhanh (streaming, cần mạng)
    python -m pipelines.load_dataset                  # full dataset, tất cả split

Output (gitignored): data/processed/<split>.parquet + data/processed/stats.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import yaml  # noqa: E402

from agent.data.clean import clean_rows, pick_text  # noqa: E402


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_row(raw: dict, cfg: dict) -> dict:
    """Map row thô -> row sạch theo configs/data.yaml."""
    text_cfg = cfg["text"]
    row = {
        "question": pick_text(raw, text_cfg["question"]["prefer"]),
        "answer": pick_text(raw, text_cfg["answer"]["prefer"]),
        "question_en": pick_text(raw, text_cfg["question_en"]["prefer"]),
        "answer_en": pick_text(raw, text_cfg["answer_en"]["prefer"]),
    }
    for col in cfg.get("keep_columns", []):
        if col in raw:
            row[col] = raw[col]
    return row


def resolve_splits(cfg: dict, dataset_id: str) -> dict[str, str]:
    """logical split ('train'/'val') -> tên split thật trên HF."""
    from datasets import get_dataset_split_names

    available = set(get_dataset_split_names(dataset_id))
    wanted: dict[str, str] = {}
    train_name = cfg["splits"]["train"]
    if train_name in available:
        wanted["train"] = train_name
    for alias in cfg["splits"]["val"]:
        if alias in available:
            wanted["val"] = alias
            break
    if not wanted:
        raise RuntimeError(f"không khớp split nào của {dataset_id}: {sorted(available)}")
    return wanted


def iter_split_rows(dataset_id: str, split: str, max_rows: int):
    """max_rows > 0: streaming lấy N row đầu; max_rows = 0: load full split.

    verification_mode="no_checks": metadata của dataset khai sai số row
    (train 106556 nhưng file thật 106557; validation 5609 vs 5608) —
    datasets 5.x chặn load nếu bật verify. Dữ liệu vẫn nguyên vẹn (Parquet).
    """
    from datasets import load_dataset

    if max_rows > 0:
        rows = load_dataset(
            dataset_id, split=split, streaming=True, verification_mode="no_checks"
        )
        for i, raw in enumerate(rows):
            if i >= max_rows:
                break
            yield raw
    else:
        yield from load_dataset(dataset_id, split=split, verification_mode="no_checks")


def process_split(
    dataset_id: str, logical: str, actual: str, cfg: dict, max_rows: int
) -> tuple[list[dict], dict]:
    cleaning = cfg["cleaning"]
    mapped = (build_row(raw, cfg) for raw in iter_split_rows(dataset_id, actual, max_rows))
    kept, stats = clean_rows(
        list(mapped),
        min_question_chars=cleaning["min_question_chars"],
        min_answer_chars=cleaning["min_answer_chars"],
    )
    return kept, {"split": actual, **stats.as_dict()}


def write_parquet(rows: list[dict], path: Path) -> None:
    import pandas as pd

    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def preview(rows: list[dict], n: int = 2, width: int = 120) -> None:
    for row in rows[:n]:
        question = row["question"].replace("\n", " ")[:width]
        answer = row["answer"].replace("\n", " ")[:width]
        print(f"  - Q: {question}")
        print(f"    A: {answer}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="HF -> clean -> Parquet + stats")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "data.yaml")
    parser.add_argument("--max-rows", type=int, default=0, help="0 = full; >0 = sample streaming")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    dataset_id = cfg["dataset_id"]
    mode = f"sample({args.max_rows})" if args.max_rows > 0 else "full"
    print(f"dataset={dataset_id} mode={mode}")

    if args.max_rows == 0:
        split_map = resolve_splits(cfg, dataset_id)
    else:
        split_map = {"train": cfg["splits"]["train"]}
    out_dir = REPO_ROOT / cfg["output"]["dir"]
    all_stats: dict[str, dict] = {"dataset_id": dataset_id, "mode": mode, "splits": {}}

    for logical, actual in split_map.items():
        kept, stats = process_split(dataset_id, logical, actual, cfg, args.max_rows)
        suffix = f"_{args.max_rows}sample" if args.max_rows > 0 else ""
        out_path = out_dir / f"{logical}{suffix}.parquet"
        write_parquet(kept, out_path)
        all_stats["splits"][logical] = stats
        print(f"[{logical} <- {actual}] {stats}")
        print(f"  -> {out_path.relative_to(REPO_ROOT)} ({len(kept)} rows)")
        if kept:
            print("  sample:")
            preview(kept)

    stats_path = out_dir / "stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(all_stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"stats -> {stats_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
