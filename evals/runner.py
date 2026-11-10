"""Eval runner — chạy scenarios A–F bằng fakes (không cần key/infra) trong CI.

Mỗi scenario YAML: question + scripted LLM responses + kỳ vọng (assertions).
CLI: python -m evals.run            (scripted, exit != 0 nếu có scenario fail)
     python -m evals.run --live     (graph THẬT — cần LLM_API_KEY + infra)

--live dùng cho evidence G1: các ca chặn trước LLM (D emergency / E injection)
assert category như scripted; còn lại in evidence thật (category/citations/answer)
vì output phụ thuộc LLM.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import yaml  # noqa: E402

from agent.fakes import FakeKg, FakeLlm, FakeRetriever, default_hits  # noqa: E402
from agent.graph import AgentAnswer, AgentGraph  # noqa: E402
from agent.retrieval.kg import Triplet  # noqa: E402
from agent.tools.builtin import default_registry  # noqa: E402

SCENARIOS_DIR = Path(__file__).resolve().parent / "scenarios"


@dataclass
class ScenarioResult:
    id: str
    category: str
    passed: bool
    errors: list[str] = field(default_factory=list)
    answer: AgentAnswer | None = None


def load_scenarios(scenarios_dir: Path | None = None) -> list[dict]:
    directory = scenarios_dir or SCENARIOS_DIR
    scenarios = []
    for path in sorted(directory.glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        data["_path"] = str(path.name)
        scenarios.append(data)
    return scenarios


def _build_graph(scenario: dict) -> tuple[AgentGraph, FakeLlm]:
    llm = FakeLlm(responses=list(scenario.get("llm") or []))
    retrieval = scenario.get("retrieval") or {}
    retriever = FakeRetriever(hits=default_hits(int(retrieval.get("hits", 2))))
    triplets = [
        Triplet(subject=t["subject"], predicate=t["predicate"], object=t["object"])
        for t in retrieval.get("triplets", [])
    ]
    kg = FakeKg(triplets=triplets)
    # tools registry THẬT (subprocess sandbox) — scenario có tool_calls chạy end-to-end
    graph = AgentGraph(llm=llm, retriever=retriever, kg=kg, tools=default_registry())
    return graph, llm


def _check_expect(expect: dict, answer: AgentAnswer, llm: FakeLlm) -> list[str]:
    errors: list[str] = []
    if "category" in expect and answer.category != expect["category"]:
        errors.append(f"category={answer.category!r}, kỳ vọng {expect['category']!r}")
    for needle in expect.get("answer_contains", []):
        if needle.lower() not in answer.text.lower():
            errors.append(f"answer thiếu chuỗi {needle!r}")
    for needle in expect.get("answer_not_contains", []):
        if needle.lower() in answer.text.lower():
            errors.append(f"answer KHÔNG được chứa {needle!r}")
    if "citations_min" in expect and len(answer.citations) < int(expect["citations_min"]):
        errors.append(f"citations={len(answer.citations)} < {expect['citations_min']}")
    if expect.get("no_llm_calls") and llm.calls:
        errors.append(f"kỳ vọng không gọi LLM nhưng đã gọi {len(llm.calls)} lần")
    if "llm_calls_exact" in expect and len(llm.calls) != int(expect["llm_calls_exact"]):
        errors.append(f"llm calls={len(llm.calls)} != {expect['llm_calls_exact']}")
    if "tool_results_min" in expect and len(answer.tool_results) < int(expect["tool_results_min"]):
        errors.append(f"tool_results={len(answer.tool_results)} < {expect['tool_results_min']}")
    for needle in expect.get("trace_contains", []):
        if not any(needle.lower() in line.lower() for line in answer.trace):
            errors.append(f"trace thiếu chuỗi {needle!r}")
    return errors


def run_scenario(scenario: dict) -> ScenarioResult:
    graph, llm = _build_graph(scenario)
    try:
        answer = graph.ask(scenario["question"], history=scenario.get("history"))
    except Exception as exc:  # noqa: BLE001 — eval không được crash cả batch
        return ScenarioResult(
            id=str(scenario.get("id", scenario.get("_path", "?"))),
            category=str(scenario.get("category", "?")),
            passed=False,
            errors=[f"exception: {type(exc).__name__}: {exc}"],
        )
    errors = _check_expect(scenario.get("expect", {}), answer, llm)
    return ScenarioResult(
        id=str(scenario.get("id", scenario.get("_path", "?"))),
        category=str(scenario.get("category", "?")),
        passed=not errors,
        errors=errors,
        answer=answer,
    )


def run_all(scenarios_dir: Path | None = None) -> list[ScenarioResult]:
    return [run_scenario(s) for s in load_scenarios(scenarios_dir)]


# Category chặn TRƯỚC LLM — kết quả tất định, assert được cả khi live
PRE_LLM_CATEGORIES = {"emergency", "blocked_injection", "blocked_pii"}


def run_live(scenarios_dir: Path | None = None) -> list[ScenarioResult]:
    """Chạy scenarios qua graph THẬT (LLM + Milvus + Neo4j từ .env).

    Assert category cho ca chặn trước LLM (tất định); các ca khác chỉ in evidence.
    """
    from agent.serve import build_default_graph  # lazy — cần deps retrieval + openai

    graph = build_default_graph()
    results: list[ScenarioResult] = []
    for scenario in load_scenarios(scenarios_dir):
        sid = str(scenario.get("id", scenario.get("_path", "?")))
        expected = scenario.get("expect", {}).get("category")
        try:
            answer = graph.ask(scenario["question"], history=scenario.get("history"))
        except Exception as exc:  # noqa: BLE001 — eval không được crash cả batch
            results.append(
                ScenarioResult(id=sid, category=str(scenario.get("category", "?")),
                               passed=False, errors=[f"exception: {type(exc).__name__}: {exc}"])
            )
            continue
        errors: list[str] = []
        if expected in PRE_LLM_CATEGORIES:
            # chặn trước LLM: text là fixed response từ constraints.yaml — tất định
            expect = scenario.get("expect", {})
            if answer.category != expected:
                errors.append(
                    f"category={answer.category!r}, kỳ vọng {expected!r} (chặn trước LLM)"
                )
            for needle in expect.get("answer_contains", []):
                if needle.lower() not in answer.text.lower():
                    errors.append(f"answer thiếu chuỗi {needle!r}")
            for needle in expect.get("answer_not_contains", []):
                if needle.lower() in answer.text.lower():
                    errors.append(f"answer KHÔNG được chứa {needle!r}")
        elif answer.category not in {
            "answered", "clarification", "refused", "emergency",
            "blocked_injection", "blocked_pii", "out_of_scope", "fallback",
        }:
            errors.append(f"category không hợp lệ: {answer.category!r}")
        results.append(
            ScenarioResult(
                id=sid,
                category=str(scenario.get("category", "?")),
                passed=not errors,
                errors=errors,
                answer=answer,
            )
        )
    return results


def main(argv: list[str] | None = None) -> int:
    args = list(argv) if argv is not None else sys.argv[1:]
    live = "--live" in args
    results = run_live() if live else run_all()
    failed = 0
    for result in results:
        mark = "PASS" if result.passed else "FAIL"
        got = result.answer.category if result.answer else "-"
        print(f"[{mark}] {result.id} ({result.category}) -> {got}")
        for error in result.errors:
            print(f"       ! {error}")
        if live and result.answer is not None:
            a = result.answer
            snippet = a.text.replace("\n", " ")[:160]
            print(f"       evidence: attempts={a.attempts} citations={len(a.citations)} "
                  f"trace_id={a.trace_id}")
            print(f"       answer: {snippet}")
        if not result.passed:
            failed += 1
    print(f"\n{len(results) - failed}/{len(results)} scenarios pass"
          f"{' (live)' if live else ''}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
