"""Agent graph — extract -> retrieve (song song) -> assemble -> generate ->
safety gate -> bounded self-correction -> tool sandbox -> validated output
(architecture.md §2).

Mọi thành phần đều inject được (LlmPort / VectorPort / KgPort / ToolRegistry) —
test & eval CI chạy bằng fakes, không cần API key hay infra; production cắm
LlmClient + MilvusVectorStore + Neo4jKgStore.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from agent.assemble.package import DraftAnswer, PromptPackage, assemble, parse_draft
from agent.extract.extractor import REFUSAL_INTENTS, Extractor, IntentEntities
from agent.ports import Citation, KgPort, LlmPort, VectorPort
from agent.safety.constraints import config_dir
from agent.safety.correction import CorrectionPolicy
from agent.safety.gate import GateVerdict, SafetyGate
from agent.tools.registry import ToolRegistry


@dataclass
class AgentAnswer:
    text: str
    # answered | clarification | refused | emergency | blocked_* | out_of_scope | fallback
    category: str
    raw_text: str = ""  # text trước khi gắn disclaimer — dùng cho history hội thoại
    citations: list[Citation] = field(default_factory=list)
    trace_id: str = ""
    attempts: int = 0
    tool_results: dict[str, dict] = field(default_factory=dict)
    dropped_sources: list[str] = field(default_factory=list)
    trace: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "answer": self.text,
            "category": self.category,
            "citations": [
                {
                    "source_row_id": c.source_row_id,
                    "chunk_index": c.chunk_index,
                    "snippet": c.snippet,
                }
                for c in self.citations
            ],
            "trace_id": self.trace_id,
            "attempts": self.attempts,
            "tool_results": self.tool_results,
            "dropped_sources": self.dropped_sources,
            "trace": self.trace,
        }


def load_agent_config(path: Path | None = None) -> dict:
    path = path or config_dir() / "agent.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


class AgentGraph:
    def __init__(
        self,
        llm: LlmPort,
        retriever: VectorPort,
        kg: KgPort,
        tools: ToolRegistry | None = None,
        gate: SafetyGate | None = None,
        max_corrections: int = 2,
        top_k: int = 5,
        kg_limit: int = 20,
        max_tool_calls_per_turn: int = 3,
    ):
        self._extractor = Extractor(llm)
        self._llm = llm
        self._retriever = retriever
        self._kg = kg
        self._tools = tools
        self._gate = gate or SafetyGate()
        self._max_corrections = max_corrections
        self._top_k = top_k
        self._kg_limit = kg_limit
        self._max_tool_calls = max_tool_calls_per_turn

    @classmethod
    def from_config(cls, llm, retriever, kg, tools=None, gate=None) -> "AgentGraph":
        cfg = load_agent_config()
        return cls(
            llm=llm,
            retriever=retriever,
            kg=kg,
            tools=tools,
            gate=gate,
            max_corrections=int(cfg.get("max_corrections", 2)),
            top_k=int(cfg.get("top_k", 5)),
            kg_limit=int(cfg.get("kg_limit", 20)),
            max_tool_calls_per_turn=int(cfg.get("max_tool_calls_per_turn", 3)),
        )

    def ask(self, question: str, history: list[dict] | None = None) -> AgentAnswer:
        """Trả lời 1 lượt. `history` = [{"role": "user"|"assistant", "content": ...}, ...]
        — turn cũ được gate lọc (injection/PII bị drop) rồi đưa vào extract + context."""
        trace_id = uuid.uuid4().hex[:12]
        trace: list[str] = [f"input: {question[:100]}"]
        fixed = self._gate.constraints.fixed_responses

        # 1) input safety — emergency/injection/PII dừng pipeline ngay
        verdict = self._gate.check_input(question)
        if not verdict.safe:
            return self._blocked(verdict, trace, trace_id)

        # 1b) history đa lượt — drop turn nghi injection/PII (fail-closed)
        kept_history, dropped_hist = self._gate.check_history(history)
        if dropped_hist:
            trace.append(f"history: drop {len(dropped_hist)} turn nghi injection/PII")

        # 2) extract intent/entities (có ngữ cảnh lịch sử)
        intent = self._extractor.extract(question, history=kept_history)
        trace.append(f"extract: intent={intent.intent} entities={intent.entities}")
        if intent.intent == "other":
            return AgentAnswer(
                text=fixed.get("out_of_scope", "Ngoài phạm vi y tế."),
                category="out_of_scope", trace_id=trace_id, trace=trace,
            )
        if intent.intent in REFUSAL_INTENTS:
            return AgentAnswer(
                text=fixed.get("refuse", "Không thể chẩn đoán/kê đơn."),
                category="refused", trace_id=trace_id, trace=trace,
            )
        if intent.needs_clarification:
            text = intent.clarification_question or fixed.get(
                "clarification", "Bạn mô tả rõ hơn được không?"
            )
            return AgentAnswer(
                text=text, category="clarification", trace_id=trace_id, trace=trace,
            )

        # 3) retrieve song song (vector + KG) — gọi tuần tự nhưng độc lập.
        # Follow-up mơ hồ ("còn ho?") thì ghép câu user trước đó vào query tìm kiếm.
        search_query = question
        if kept_history:
            last_user = next(
                (t["content"] for t in reversed(kept_history) if t["role"] == "user"), ""
            )
            if last_user:
                search_query = f"{last_user} {question}"
        hits = self._retriever.search(search_query, top_k=self._top_k)
        triplets = self._kg.query(intent.entities, limit=self._kg_limit)
        hits, triplets, dropped = self._gate.filter_sources(hits, triplets)
        trace.append(f"retrieve: {len(hits)} hits, {len(triplets)} triplets, drop {len(dropped)}")

        # 4) generate + safety gate với self-correction bounded
        policy = CorrectionPolicy(max_rounds=self._max_corrections)
        answer_text: str | None = None
        attempts = 0
        tool_results: dict[str, dict] = {}
        while True:
            draft, gen_attempts, tool_results = self._generate_with_tools(
                question, intent, hits, triplets, policy.last_reason, kept_history
            )
            attempts += gen_attempts
            verdict = self._gate.check_output(draft.answer)
            if verdict.safe:
                answer_text = draft.answer
                trace.append(f"gate: ok (after {attempts} generates)")
                break
            policy.record(verdict.reason, verdict.category)
            trace.append(
                f"gate.fail[round {policy.rounds_used}]: {verdict.category} — {verdict.reason}"
            )
            if not policy.should_retry():
                trace.append("correction: hết vòng -> fallback")
                break

        if answer_text is None:
            return AgentAnswer(
                text=fixed.get("fallback", "Không thể trả lời an toàn."),
                category="fallback",
                trace_id=trace_id,
                attempts=attempts,
                tool_results=tool_results,
                dropped_sources=dropped,
                trace=trace,
            )

        disclaimer = fixed.get("disclaimer", "").strip()
        raw_answer = answer_text
        if disclaimer:
            answer_text = f"{answer_text}\n\n{disclaimer}"
        return AgentAnswer(
            text=answer_text,
            category="answered",
            raw_text=raw_answer,
            citations=self._citations(hits),
            trace_id=trace_id,
            attempts=attempts,
            tool_results=tool_results,
            dropped_sources=dropped,
            trace=trace,
        )

    def _blocked(self, verdict: GateVerdict, trace: list[str], trace_id: str) -> AgentAnswer:
        fixed = self._gate.constraints.fixed_responses
        if verdict.category == "emergency":
            text = fixed.get("emergency", "Gọi cấp cứu 115 ngay.")
        elif verdict.category == "pii":
            text = fixed.get("blocked_pii", "Không chia sẻ thông tin cá nhân.")
        else:
            text = fixed.get("blocked_injection", "Yêu cầu không hợp lệ.")
        trace.append(f"input.gate: {verdict.category} — {verdict.reason}")
        category = "emergency" if verdict.category == "emergency" else f"blocked_{verdict.category}"
        return AgentAnswer(
            text=text,
            category=category,
            trace_id=trace_id,
            trace=trace,
        )

    def _generate_with_tools(
        self,
        query: str,
        intent: IntentEntities,
        hits,
        triplets,
        repair_reason: str,
        history: list[dict] | None = None,
    ) -> tuple[DraftAnswer, int, dict[str, dict]]:
        """Assemble -> generate; nếu draft gọi tool thì chạy sandbox và fold 1 lần."""
        tool_results: dict[str, dict] = {}
        pkg: PromptPackage = assemble(
            query, intent, hits, triplets, repair_reason=repair_reason, history=history
        )
        raw = self._llm.complete_json(pkg.system, pkg.user)
        draft = parse_draft(raw)
        attempts = 1
        if draft.tool_calls and self._tools is not None:
            for call in draft.tool_calls[: self._max_tool_calls]:
                result = self._tools.run(call["name"], call["args"])
                tool_results[f"{call['name']}#{len(tool_results)}"] = result.as_dict()
            pkg = assemble(
                query, intent, hits, triplets,
                tool_results=tool_results, repair_reason=repair_reason, history=history,
            )
            draft = parse_draft(self._llm.complete_json(pkg.system, pkg.user))
            draft.tool_calls = []  # bounded: chỉ fold tool 1 lần
            attempts += 1
        return draft, attempts, tool_results

    @staticmethod
    def _citations(hits, limit: int = 3) -> list[Citation]:
        return [
            Citation(
                source_row_id=h.source_row_id,
                chunk_index=h.chunk_index,
                snippet=h.text[:120],
            )
            for h in hits[:limit]
        ]
