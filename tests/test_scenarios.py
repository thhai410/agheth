"""Chạy tất cả eval scenarios A–F (YAML) như regression suite trong CI (M4)."""
import pytest

from agent.fakes import FakeLlm
from agent.graph import AgentAnswer
from evals.runner import _check_expect, load_scenarios, run_scenario

SCENARIOS = load_scenarios()


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s["id"] for s in SCENARIOS])
def test_scenario(scenario):
    result = run_scenario(scenario)
    assert result.passed, "; ".join(result.errors)


def test_all_categories_covered():
    categories = {s["category"] for s in SCENARIOS}
    assert {"A", "B", "C", "D", "E", "F"} <= categories


def test_trace_contains_assertion():
    # guard: key trace_contains phải thật sự được _check_expect kiểm (không bị ignore)
    answer = AgentAnswer(
        text="ok", category="answered", trace=["history: drop 1 turn nghi injection/PII"]
    )
    llm = FakeLlm(responses=[])
    assert not _check_expect({"trace_contains": ["history: drop 1"]}, answer, llm)
    assert _check_expect({"trace_contains": ["không có trong trace"]}, answer, llm)
