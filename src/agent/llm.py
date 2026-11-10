"""LLM client — endpoint OpenAI-compatible (DeepSeek mặc định; Gemini/endpoint khác
chỉ cần đổi base_url + model, xem .env.example).

Dùng cho triplet extraction (M1), intent/entity extraction + grader (M2).
Key lấy từ Settings (env/.env) — không hardcode, không commit.
"""
from __future__ import annotations

import json
import re

# ```json ... ``` hoặc ``` ... ``` — nhiều model (vd Gemini) bọc JSON trong code fence
_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)

# Lỗi tầng mạng/auth/rate-limit/5xx — retry vô nghĩa, khác với "endpoint từ chối response_format".
# 5xx (vd 503 ServiceUnavailable của Gemini free tier) cũng raise thẳng: fallback
# không response_format chỉ nhân đôi request vào một endpoint đang lỗi.
# (BadRequestError KHÔNG nằm trong list — đó là case fallback thật sự.)
_TRANSPORT_ERROR_NAMES = (
    "timeout",
    "connection",
    "connecterror",
    "ratelimit",
    "authentication",
    "permission",
    "apierror",
    "statuserror",  # apistatuserror / internalservererror — lỗi HTTP từ SDK
    "internalservererror",
    "serviceunavailable",
)


def _is_transport_error(exc: Exception) -> bool:
    name = type(exc).__name__.lower()
    return any(k in name for k in _TRANSPORT_ERROR_NAMES)


class LlmClient:
    """Thin wrapper quanh OpenAI SDK — mọi endpoint OpenAI-compatible."""

    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-v4.1",
        base_url: str = "https://api.deepseek.com",
        timeout: float = 60.0,
    ):
        if not api_key:
            raise RuntimeError("thiếu LLM_API_KEY (hoặc DEEPSEEK_API_KEY) — đặt trong .env")
        from openai import OpenAI  # lazy — chỉ cần khi thật sự gọi LLM

        # timeout chặn request treo vô hạn (endpoint rate-limit có thể "ngậm" request)
        self._client = OpenAI(
            api_key=api_key, base_url=base_url, timeout=timeout, max_retries=1
        )
        self.model = model

    def complete_json(self, system: str, user: str, *, temperature: float = 0.0) -> dict:
        """Gọi chat completion, kỳ vọng output JSON object, trả về dict đã parse.

        Một số endpoint OpenAI-compatible (vd Gemini compat) chưa chắc hỗ trợ
        `response_format: json_object` — nếu bị từ chối thì gọi lại không có
        tham số đó và tự trích JSON từ text.
        """
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        try:
            content = self._create(messages, temperature, json_mode=True)
        except Exception as e:  # noqa: BLE001 — endpoint không hỗ trợ response_format thì fallback
            if _is_transport_error(e):
                raise  # timeout/mạng/auth/rate-limit — gọi lại chỉ làm chậm gấp đôi
            content = self._create(messages, temperature, json_mode=False)
        return _extract_json(content)

    def _create(self, messages: list[dict], temperature: float, *, json_mode: bool) -> str:
        kwargs: dict = dict(
            model=self.model,
            messages=messages,
            temperature=temperature,
        )
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = self._client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""


# Tên cũ — giữ lại để code/ghi chú cũ không vỡ
DeepSeekClient = LlmClient


def _extract_json(content: str) -> dict:
    """Parse JSON object từ output LLM — chịu được code fence và text thừa quanh JSON."""
    text = (content or "").strip()
    if not text:
        raise ValueError("LLM trả về rỗng — không có JSON để parse")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    fence = _FENCE_RE.search(text)
    if fence:
        try:
            return json.loads(fence.group(1))
        except json.JSONDecodeError:
            pass
    # JSON nằm lẫn trong text: lấy từ { đầu tiên tới } cuối cùng
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    raise ValueError(f"không parse được JSON từ output LLM: {text[:200]!r}")
