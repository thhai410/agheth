"""Context assembly — DSH skill (architecture.md §3–4). Hàm thuần, không gọi LLM."""
from agent.assemble.package import (
    ALLOWED_TOOL_HINTS,
    SYSTEM_PROMPT,
    DraftAnswer,
    PromptPackage,
    assemble,
    parse_draft,
)

__all__ = [
    "ALLOWED_TOOL_HINTS",
    "SYSTEM_PROMPT",
    "DraftAnswer",
    "PromptPackage",
    "assemble",
    "parse_draft",
]
