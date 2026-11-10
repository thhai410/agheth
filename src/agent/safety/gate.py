"""Safety gate — "Is Execution Safe?" (architecture.md §2–3, §7).

Fail-closed: không rõ ràng => chặn. Rules keyword trước; LLM grader là hook
tùy chọn (mặc định None — CI chạy rules thuần, không cần key).
"""
from __future__ import annotations

from dataclasses import dataclass

from agent.retrieval.kg import Triplet
from agent.retrieval.vector import Hit
from agent.safety.constraints import (
    Constraints,
    find_diagnosis,
    find_emergency,
    find_injection,
    find_pii,
    find_prescription,
    load_constraints,
)


@dataclass(frozen=True)
class GateVerdict:
    safe: bool
    reason: str = ""
    category: str = ""  # emergency | injection | pii | diagnosis | prescription | empty_answer

    @staticmethod
    def ok() -> "GateVerdict":
        return GateVerdict(safe=True)


class SafetyGate:
    def __init__(
        self,
        constraints: Constraints | None = None,
        llm_grader=None,  # optional LlmPort — chấm output nâng cao (không bắt buộc)
    ):
        self._c = constraints or load_constraints()
        self._llm_grader = llm_grader

    @property
    def constraints(self) -> Constraints:
        return self._c

    def check_input(self, query: str) -> GateVerdict:
        """Kiểm input người dùng: cấp cứu -> verdict riêng; injection/PII -> chặn."""
        match = find_emergency(query, self._c)
        if match:
            return GateVerdict(False, f"từ khóa cấp cứu: {match!r}", "emergency")
        match = find_injection(query, self._c)
        if match:
            return GateVerdict(False, f"prompt injection: {match!r}", "injection")
        match = find_pii(query, self._c)
        if match:
            return GateVerdict(False, f"chứa PII (pattern {match!r})", "pii")
        return GateVerdict.ok()

    def check_history(self, history: list[dict]) -> tuple[list[dict], list[str]]:
        """Lọc lịch sử hội thoại trước khi đưa vào context — turn có injection/PII bị drop.

        Fail-closed: turn lạ (role không phải user/assistant) cũng bị loại.
        Trả về (kept, dropped_log).
        """
        kept: list[dict] = []
        dropped: list[str] = []
        for i, turn in enumerate(history or []):
            role = turn.get("role") if isinstance(turn, dict) else None
            content = turn.get("content", "") if isinstance(turn, dict) else ""
            if role not in ("user", "assistant") or not str(content).strip():
                dropped.append(f"history[{i}]: turn không hợp lệ (role={role!r})")
                continue
            match = find_injection(str(content), self._c) or find_pii(str(content), self._c)
            if match:
                dropped.append(f"history[{i}] role={role}: {match!r}")
            else:
                kept.append({"role": role, "content": str(content)})
        return kept, dropped

    def filter_sources(
        self, hits: list[Hit], triplets: list[Triplet]
    ) -> tuple[list[Hit], list[Triplet], list[str]]:
        """Loại tài liệu nghi injection (drop + log, fail-closed)."""
        kept_hits, dropped = [], []
        for hit in hits:
            match = find_injection(hit.text, self._c) or find_injection(hit.question, self._c)
            if match:
                dropped.append(f"hit src={hit.source_row_id}#{hit.chunk_index}: {match!r}")
            else:
                kept_hits.append(hit)
        kept_triplets = [
            t for t in triplets
            if not (find_injection(t.subject, self._c) or find_injection(t.object, self._c))
        ]
        return kept_hits, kept_triplets, dropped

    def check_output(self, answer: str) -> GateVerdict:
        """Kiểm draft answer: chẩn đoán/kê đơn/injection/PII/answer rỗng."""
        if not answer.strip():
            return GateVerdict(False, "answer rỗng", "empty_answer")
        match = find_diagnosis(answer, self._c)
        if match:
            return GateVerdict(False, f"output chẩn đoán: {match!r}", "diagnosis")
        match = find_prescription(answer, self._c)
        if match:
            return GateVerdict(False, f"output kê đơn: {match!r}", "prescription")
        match = find_injection(answer, self._c)
        if match:
            return GateVerdict(False, f"output chứa injection: {match!r}", "injection")
        match = find_pii(answer, self._c)
        if match:
            return GateVerdict(False, f"output phản hồi PII: {match!r}", "pii")
        if self._llm_grader is not None:
            verdict = self._llm_grade(answer)
            if verdict is not None and not verdict.safe:
                return verdict
        return GateVerdict.ok()

    def _llm_grade(self, answer: str) -> GateVerdict | None:
        """LLM grader hook — fail-closed: response không parse được -> chặn."""
        try:
            raw = self._llm_grader.complete_json(
                "Chấm câu trả lời y tế: JSON {\"safe\": bool, \"reason\": str}.",
                answer,
            )
            safe = bool(raw.get("safe", False))
            reason = str(raw.get("reason", ""))
        except Exception as exc:  # noqa: BLE001 — fail-closed
            return GateVerdict(False, f"LLM grader lỗi: {exc}", "grader_error")
        if safe:
            return GateVerdict.ok()
        return GateVerdict(False, reason or "grader: unsafe", "grader")
