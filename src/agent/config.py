"""Runtime settings đọc từ environment / .env.

Secrets KHÔNG commit (.env nằm trong .gitignore) và không gửi trong chat.

LLM là endpoint OpenAI-compatible: mặc định DeepSeek, đổi sang Gemini hay endpoint
khác chỉ cần set LLM_BASE_URL + LLM_MODEL + LLM_API_KEY (xem .env.example) —
không cần sửa code.
"""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM — endpoint OpenAI-compatible (extract / assemble / safety grader, M2)
    llm_api_key: str = ""
    llm_base_url: str = "https://api.deepseek.com"
    llm_model: str = ""

    # Tên cũ — vẫn đọc được từ .env cũ (llm_* ở trên được ưu tiên nếu cùng set)
    deepseek_api_key: str = ""
    deepseek_model: str = ""

    # Retrieval (M1)
    milvus_uri: str = "http://localhost:19530"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""

    # DeepSeek Harness runtime (M2) — bản pin nằm trong lockfile
    dsh_version: str = ""

    @property
    def api_key(self) -> str:
        return self.llm_api_key or self.deepseek_api_key

    @property
    def model_id(self) -> str:
        return self.llm_model or self.deepseek_model or "deepseek-v4.1"


def get_settings() -> Settings:
    return Settings()
