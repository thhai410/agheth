"""Tool registry + sandbox — whitelist + validate tham số + subprocess cách ly
(architecture.md §8).

Fail-closed: tool ngoài whitelist -> từ chối (KHÔNG âm thầm bỏ qua); args sai
schema -> từ chối; subprocess có timeout — crash/timeout không sập agent.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[3]
MAX_STR_LEN = 500
MAX_LIST_ITEMS = 20


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    params: dict[str, str]  # param -> "str" | "int" | "list"
    timeout_s: float = 5.0


@dataclass
class ToolResult:
    name: str
    ok: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: str = ""

    def as_dict(self) -> dict[str, Any]:
        if self.ok:
            return {"ok": True, "data": self.data}
        return {"ok": False, "error": self.error}


def validate_args(spec: ToolSpec, args: dict[str, Any]) -> str | None:
    """Trả về lý do lỗi, hoặc None nếu hợp lệ (fail-closed)."""
    for key in args:
        if key not in spec.params:
            return f"tham số không có trong schema: {key!r}"
    for key, kind in spec.params.items():
        if key not in args:
            return f"thiếu tham số bắt buộc: {key!r}"
        value = args[key]
        if kind == "str":
            if not isinstance(value, str) or not value.strip():
                return f"{key!r} phải là chuỗi khác rỗng"
            if len(value) > MAX_STR_LEN:
                return f"{key!r} quá dài (> {MAX_STR_LEN})"
        elif kind == "int":
            if not isinstance(value, int) or isinstance(value, bool):
                return f"{key!r} phải là số nguyên"
        elif kind == "list":
            if not isinstance(value, list) or not value:
                return f"{key!r} phải là danh sách khác rỗng"
            if len(value) > MAX_LIST_ITEMS:
                return f"{key!r} quá dài (> {MAX_LIST_ITEMS} phần tử)"
            if not all(isinstance(item, str) and item.strip() for item in value):
                return f"{key!r} phải là danh sách chuỗi khác rỗng"
        else:
            return f"schema không rõ kiểu {kind!r} của {key!r}"
    return None


class SubprocessExecutor:
    """Chạy tool trong subprocess cách ly (timeout + crash isolation)."""

    def __init__(self, timeout_s: float | None = None):
        self._timeout_s = timeout_s

    def __call__(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        timeout = self._timeout_s
        payload = json.dumps({"name": name, "args": args}, ensure_ascii=False)
        env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "agent.tools._runner", payload],
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": f"tool {name!r} timeout sau {timeout}s"}
        if proc.returncode != 0:
            return {"ok": False, "error": f"tool {name!r} crash: {proc.stderr[-300:]}"}
        try:
            return json.loads(proc.stdout.strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError):
            return {"ok": False, "error": f"tool {name!r} output không phải JSON"}


class ToolRegistry:
    """Whitelist tool + validate args + thực thi qua executor (sandbox hoặc direct)."""

    def __init__(
        self,
        specs: list[ToolSpec],
        executor: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
        default_timeout_s: float = 5.0,
    ):
        self._specs = {spec.name: spec for spec in specs}
        self._executor = executor or SubprocessExecutor(timeout_s=default_timeout_s)

    def names(self) -> list[str]:
        return sorted(self._specs)

    def run(self, name: str, args: dict[str, Any]) -> ToolResult:
        spec = self._specs.get(name)
        if spec is None:
            return ToolResult(name=name, ok=False, error=f"tool không có trong whitelist: {name!r}")
        error = validate_args(spec, args if isinstance(args, dict) else {})
        if error:
            return ToolResult(name=name, ok=False, error=f"args không hợp lệ: {error}")
        try:
            raw = self._executor(name, dict(args))
        except Exception as exc:  # noqa: BLE001 — cách ly lỗi tool
            return ToolResult(name=name, ok=False, error=f"tool lỗi: {exc}")
        if not isinstance(raw, dict):
            return ToolResult(name=name, ok=False, error="tool output không phải dict")
        if not raw.get("ok", False):
            return ToolResult(name=name, ok=False, error=str(raw.get("error", "tool fail")))
        data = raw.get("data", {})
        if not isinstance(data, dict):
            data = {"result": data}
        return ToolResult(name=name, ok=True, data=data)
