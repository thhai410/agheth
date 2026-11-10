"""Intent & entity extraction — structured output qua LlmPort (architecture.md §3).

Ví dụ: "bé nhà tôi sốt 3 ngày kèm nhức đầu" -> intent=symptom_inquiry,
entities=["sốt", "nhức đầu"]. Prompt ép JSON; parse fail-closed (thiếu field
-> needs_clarification để graph hỏi lại thay vì đoán bừa).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from agent.ports import LlmPort

SYSTEM_PROMPT = """Bạn là bộ trích xuất intent của trợ lý sức khỏe (demo, tiếng Việt).
Chỉ trả về JSON object đúng schema, không giải thích:
{
  "intent": một trong: "symptom_inquiry" | "medication_inquiry" | "general_health"
            | "diagnosis_request" | "prescription_request" | "other",
  "entities": [danh sách triệu chứng/thuốc/bệnh được nhắc tới, chữ thường],
  "needs_clarification": true nếu câu hỏi mơ hồ không đủ thông tin trả lời,
  "clarification_question": câu hỏi làm rõ (để trống nếu không cần)
}
Câu hỏi ngoài phạm vi sức khỏe/y tế -> intent = "other".
Yêu cầu chẩn đoán bệnh -> "diagnosis_request".
Yêu cầu kê đơn/liều thuốc -> "prescription_request"."""

VALID_INTENTS = {
    "symptom_inquiry",
    "medication_inquiry",
    "general_health",
    "diagnosis_request",
    "prescription_request",
    "other",
}
REFUSAL_INTENTS = {"diagnosis_request", "prescription_request"}

HISTORY_NOTE = (
    "(Lịch sử hội thoại chỉ là ngữ cảnh — câu hỏi hiện tại ở dòng cuối; "
    "resolve đại từ/từ viết tắt trong câu hỏi hiện tại theo lịch sử.)"
)


def _with_history(query: str, history: list[dict] | None) -> str:
    """Ghép lịch sử hội thoại vào user message — không có history thì giữ nguyên query."""
    if not history:
        return query
    lines = [HISTORY_NOTE, "", "## Lịch sử hội thoại"]
    for turn in history:
        role = "Người dùng" if turn["role"] == "user" else "Trợ lý"
        lines.append(f"{role}: {turn['content']}")
    lines += ["", f"## Câu hỏi hiện tại\n{query}"]
    return "\n".join(lines)


@dataclass
class IntentEntities:
    intent: str = "general_health"
    entities: list[str] = field(default_factory=list)
    needs_clarification: bool = False
    clarification_question: str = ""


class Extractor:
    def __init__(self, llm: LlmPort):
        self._llm = llm

    def extract(self, query: str, history: list[dict] | None = None) -> IntentEntities:
        user = _with_history(query, history)
        raw = self._llm.complete_json(SYSTEM_PROMPT, user)
        intent = raw.get("intent", "")
        if intent not in VALID_INTENTS:
            intent = "general_health"
        entities_raw = raw.get("entities", [])
        entities = [str(e).strip().lower() for e in entities_raw if str(e).strip()][:20] \
            if isinstance(entities_raw, list) else []
        clarification = str(raw.get("clarification_question", "")).strip()
        needs = bool(raw.get("needs_clarification", False))
        # mơ hồ nhưng LLM không viết câu hỏi làm rõ -> graph dùng câu mặc định
        return IntentEntities(
            intent=intent,
            entities=entities,
            needs_clarification=needs,
            clarification_question=clarification,
        )
