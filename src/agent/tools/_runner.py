"""Entry point cho subprocess sandbox — nhận JSON payload trên argv, in JSON kết quả.

Payload: {"name": "...", "args": {...}} — CHỈ chạy tool trong whitelist built-in.
"""
from __future__ import annotations

import json
import sys

from agent.tools.builtin import drug_lookup, symptom_checker

_RUNNERS = {
    "drug_lookup": lambda args: drug_lookup(args["name"]),
    "symptom_checker": lambda args: symptom_checker(args["symptoms"]),
}


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        print(json.dumps({"ok": False, "error": "thiếu payload"}))
        return 1
    try:
        payload = json.loads(argv[0])
        name = payload["name"]
        args = payload["args"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        print(json.dumps({"ok": False, "error": f"payload không hợp lệ: {exc}"}))
        return 1
    runner = _RUNNERS.get(name)
    if runner is None:
        print(json.dumps({"ok": False, "error": f"tool không có trong whitelist: {name!r}"}))
        return 1
    try:
        data = runner(args)
        print(json.dumps({"ok": True, "data": data}, ensure_ascii=False))
        return 0
    except Exception as exc:  # noqa: BLE001 — in lỗi ra JSON, không sập subprocess
        payload_out = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(payload_out, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
