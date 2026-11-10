"""Tool sandbox: whitelist fail-closed + validate args + subprocess cách ly (M3)."""
import pytest

from agent.fakes import FakeKg
from agent.tools.builtin import default_registry, drug_lookup, symptom_checker
from agent.tools.registry import ToolRegistry, ToolSpec, validate_args

SPEC = ToolSpec(name="demo", description="demo", params={"name": "str", "limit": "int"})


def test_validate_args_ok():
    assert validate_args(SPEC, {"name": "paracetamol", "limit": 3}) is None


def test_validate_args_fail_closed():
    assert validate_args(SPEC, {"name": "x"})  # thiếu limit
    assert validate_args(SPEC, {"name": "x", "limit": 1, "hack": True})  # thừa tham số
    assert validate_args(SPEC, {"name": "", "limit": 1})  # chuỗi rỗng
    assert validate_args(SPEC, {"name": "x" * 600, "limit": 1})  # quá dài
    assert validate_args(SPEC, {"name": "x", "limit": "3"})  # sai kiểu
    assert validate_args(SPEC, {"name": "x", "limit": True})  # bool không phải int


def test_registry_rejects_unknown_tool():
    registry = ToolRegistry([SPEC], executor=lambda name, args: {"ok": True, "data": {}})
    result = registry.run("evil_tool", {"name": "x", "limit": 1})
    assert not result.ok and "whitelist" in result.error


def test_registry_direct_executor():
    registry = ToolRegistry([SPEC], executor=lambda name, args: {"ok": True, "data": args})
    result = registry.run("demo", {"name": "paracetamol", "limit": 2})
    assert result.ok and result.data == {"name": "paracetamol", "limit": 2}


def test_registry_isolates_executor_crash():
    def boom(name, args):
        raise RuntimeError("tool crash")

    registry = ToolRegistry([SPEC], executor=boom)
    result = registry.run("demo", {"name": "x", "limit": 1})
    assert not result.ok and "tool crash" in result.error


def test_drug_lookup_table():
    found = drug_lookup("Paracetamol")
    assert found["found"] and found["class"]
    assert drug_lookup("thuốc không tồn tại xyz")["found"] is False


def test_symptom_checker_uses_kg_not_llm():
    kg = FakeKg()
    result = symptom_checker(["sốt", "nhức đầu"], kg=kg)
    assert "cảm cúm" in result["related_conditions"]
    assert result["evidence"] and kg.calls


@pytest.mark.slow
def test_subprocess_sandbox_drug_lookup():
    """Chạy tool THẬT qua subprocess cách ly — timeout + crash isolation."""
    registry = default_registry()
    result = registry.run("drug_lookup", {"name": "ibuprofen"})
    assert result.ok and result.data["found"] is True
