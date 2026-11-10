"""Fakes cho test + eval CI — cùng contract với ports.py, không cần infra/key.

FakeLlm trả scripted response theo thứ tự queue (mỗi lượt gọi pop 1 phần tử).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent.retrieval.kg import Triplet
from agent.retrieval.vector import Hit


class FakeLlm:
    """Scripted LLM — queue response JSON theo thứ tự gọi. Ghi lại mọi lời gọi."""

    def __init__(self, responses: list[dict] | None = None, default: dict | None = None):
        self._queue = list(responses or [])
        self._default = default
        self.calls: list[dict[str, Any]] = []

    def complete_json(self, system: str, user: str, *, temperature: float = 0.0) -> dict:
        self.calls.append({"system": system, "user": user, "temperature": temperature})
        if self._queue:
            return self._queue.pop(0)
        if self._default is not None:
            return self._default
        raise AssertionError("FakeLlm: hết scripted response (kiểm tra số lần gọi LLM)")


class FakeRetriever:
    """Trả danh sách Hit cố định — mô phỏng Milvus hybrid search."""

    def __init__(self, hits: list[Hit] | None = None):
        self._hits = hits if hits is not None else default_hits()
        self.calls: list[str] = []

    def search(self, query: str, top_k: int = 5) -> list[Hit]:
        self.calls.append(query)
        return self._hits[:top_k]


class FakeKg:
    """Trả triplet cố định — mô phỏng Neo4j query."""

    def __init__(self, triplets: list[Triplet] | None = None):
        self._triplets = triplets if triplets is not None else default_triplets()
        self.calls: list[list[str]] = []

    def query(self, entities: list[str], limit: int = 20) -> list[Triplet]:
        self.calls.append(list(entities))
        return self._triplets[:limit]


@dataclass
class FakeToolExecutor:
    """Thực thi tool trực tiếp với kết quả scripted — key = tên tool."""

    results: dict[str, dict[str, Any]] = field(default_factory=dict)
    calls: list[tuple[str, dict]] = field(default_factory=list)

    def __call__(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((name, dict(args)))
        return self.results.get(name, {"ok": True, "data": {}})


def default_hits(n: int = 2) -> list[Hit]:
    return [
        Hit(
            text=f"Hỏi: mẫu câu hỏi sức khỏe số {i}. Đáp: câu trả lời tham khảo của bác sĩ số {i}.",
            score=0.5 + i * 0.01,
            source_row_id=f"{1000 + i}",
            chunk_index=0,
            question=f"mẫu câu hỏi sức khỏe số {i}",
        )
        for i in range(n)
    ]


def default_triplets() -> list[Triplet]:
    return [
        Triplet(subject="cảm cúm", predicate="has_symptom", object="sốt"),
        Triplet(subject="cảm cúm", predicate="has_symptom", object="nhức đầu"),
    ]
