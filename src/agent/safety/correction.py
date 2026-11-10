"""Self-correction loop — bounded, max 2 vòng (architecture.md §2, §7).

Gate fail -> trả lý do cho Context Assembly, thử lại; quá số vòng -> fallback
an toàn + log trace. Mọi lần fail đều được ghi lại trong `CorrectionPolicy`.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CorrectionAttempt:
    round: int
    reason: str
    category: str = ""


@dataclass
class CorrectionPolicy:
    max_rounds: int = 2
    attempts: list[CorrectionAttempt] = field(default_factory=list)

    @property
    def rounds_used(self) -> int:
        return len(self.attempts)

    def should_retry(self) -> bool:
        return self.rounds_used <= self.max_rounds

    def record(self, reason: str, category: str = "") -> CorrectionAttempt:
        attempt = CorrectionAttempt(
            round=self.rounds_used + 1, reason=reason, category=category
        )
        self.attempts.append(attempt)
        return attempt

    @property
    def last_reason(self) -> str:
        return self.attempts[-1].reason if self.attempts else ""
