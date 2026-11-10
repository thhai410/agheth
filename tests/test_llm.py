"""LLM client: trích JSON (fence/text fallback) + fallback response_format (Gemini compat)."""
from types import SimpleNamespace

import pytest

from agent.config import Settings
from agent.llm import DeepSeekClient, LlmClient, _extract_json


def test_extract_json_plain():
    assert _extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_fenced():
    assert _extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert _extract_json('```\n{"a": 1}\n```') == {"a": 1}


def test_extract_json_embedded_in_text():
    assert _extract_json('Kết quả: {"a": {"b": 2}} nhé') == {"a": {"b": 2}}


def test_extract_json_errors():
    with pytest.raises(ValueError):
        _extract_json("")
    with pytest.raises(ValueError):
        _extract_json("xin lỗi, không có JSON ở đây")


def _make_llm(content: str, reject_json_mode: bool = False):
    """LlmClient với fake OpenAI client — không gọi mạng."""
    calls: list[dict] = []

    def create(**kwargs):
        calls.append(kwargs)
        if reject_json_mode and "response_format" in kwargs:
            raise RuntimeError("response_format not supported")  # mô phỏng Gemini compat
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )

    llm = LlmClient(api_key="test-key", model="m")
    llm._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return llm, calls


def test_complete_json_uses_response_format_when_supported():
    llm, calls = _make_llm('{"answer": "ok"}')
    assert llm.complete_json("s", "u") == {"answer": "ok"}
    assert calls[0]["response_format"] == {"type": "json_object"}
    assert len(calls) == 1


def test_complete_json_falls_back_without_response_format():
    llm, calls = _make_llm('```json\n{"answer": "ok"}\n```', reject_json_mode=True)
    assert llm.complete_json("s", "u") == {"answer": "ok"}
    assert len(calls) == 2
    assert "response_format" in calls[0] and "response_format" not in calls[1]


def test_complete_json_transport_error_not_retried():
    # timeout/rate-limit phải nổi thẳng — retry không response_format chỉ làm chậm gấp đôi
    calls: list[dict] = []

    def create(**kwargs):
        calls.append(kwargs)
        raise TimeoutError("request timed out")

    llm = LlmClient(api_key="test-key", model="m")
    llm._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    with pytest.raises(TimeoutError):
        llm.complete_json("s", "u")
    assert len(calls) == 1


def test_complete_json_server_error_not_retried():
    # 5xx (vd 503 ServiceUnavailable) là lỗi server — fallback không response_format
    # chỉ nhân đôi request vào endpoint đang lỗi
    class InternalServerError(Exception):
        pass

    calls: list[dict] = []

    def create(**kwargs):
        calls.append(kwargs)
        raise InternalServerError("Error code: 503 - Service Unavailable")

    llm = LlmClient(api_key="test-key", model="m")
    llm._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    with pytest.raises(InternalServerError):
        llm.complete_json("s", "u")
    assert len(calls) == 1


def test_deepseek_client_alias_still_works():
    assert DeepSeekClient is LlmClient


def test_settings_llm_alias_precedence():
    # init kwargs ghi đè env/.env — test không phụ thuộc secrets của user
    s = Settings(
        _env_file=None,
        llm_api_key="new-key", deepseek_api_key="old-key",
        llm_model="gemini-3.8-flash", deepseek_model="deepseek-v4.1",
    )
    assert s.api_key == "new-key"
    assert s.model_id == "gemini-3.8-flash"

    legacy = Settings(
        _env_file=None,
        llm_api_key="", deepseek_api_key="old-key",
        llm_model="", deepseek_model="deepseek-chat",
    )
    assert legacy.api_key == "old-key"
    assert legacy.model_id == "deepseek-chat"

    empty = Settings(
        _env_file=None,
        llm_api_key="", deepseek_api_key="", llm_model="", deepseek_model="",
    )
    assert empty.api_key == ""
    assert empty.model_id == "deepseek-v4.1"  # default khi không set gì
