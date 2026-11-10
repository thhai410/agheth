"""Adapter interfaces (architecture.md §4) — mọi lời gọi ra ngoài đi qua đây.

DSH dev-preview có breaking change thì chỉ lớp orchestration phải vá;
các implementation thật (LlmClient, MilvusVectorStore, Neo4jKgStore)
và fakes đều dùng chung contract này (structural typing — không cần inherit).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from agent.retrieval.kg import Triplet
from agent.retrieval.vector import Hit


@dataclass(frozen=True)
class Citation:
    source_row_id: str
    chunk_index: int
    snippet: str


@runtime_checkable
class LlmPort(Protocol):
    def complete_json(self, system: str, user: str, *, temperature: float = 0.0) -> dict: ...


@runtime_checkable
class VectorPort(Protocol):
    def search(self, query: str, top_k: int = 5) -> list[Hit]: ...


@runtime_checkable
class KgPort(Protocol):
    def query(self, entities: list[str], limit: int = 20) -> list[Triplet]: ...


@runtime_checkable
class ToolExecutorPort(Protocol):
    """Thực thi 1 tool với args đã validate — sandbox subprocess hoặc direct (test)."""

    def __call__(self, name: str, args: dict[str, Any]) -> dict[str, Any]: ...
