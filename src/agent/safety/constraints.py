"""Ràng buộc an toàn (tiếng Việt), fail-closed — architecture.md §7.

Nội dung keyword/pattern nằm ở configs/agent.yaml (đổi = HITL G0 contract MR).
Match substring không phân biệt hoa thường; PII = regex.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import yaml


def config_dir() -> Path:
    """Thư mục configs/ — ưu tiên AGENT_CONFIG_DIR, rồi repo root, rồi cwd."""
    env = os.environ.get("AGENT_CONFIG_DIR")
    if env:
        return Path(env)
    repo_configs = Path(__file__).resolve().parents[3] / "configs"
    if repo_configs.is_dir():
        return repo_configs
    return Path.cwd() / "configs"


@dataclass(frozen=True)
class Constraints:
    emergency_keywords: tuple[str, ...]
    diagnosis_patterns: tuple[str, ...]
    prescription_patterns: tuple[str, ...]
    injection_patterns: tuple[str, ...]
    pii_patterns: tuple[str, ...]
    fixed_responses: dict[str, str]


def load_constraints(path: Path | None = None) -> Constraints:
    path = path or config_dir() / "constraints.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return Constraints(
        emergency_keywords=tuple(data.get("emergency_keywords", ())),
        diagnosis_patterns=tuple(data.get("diagnosis_patterns", ())),
        prescription_patterns=tuple(data.get("prescription_patterns", ())),
        injection_patterns=tuple(data.get("injection_patterns", ())),
        pii_patterns=tuple(data.get("pii_patterns", ())),
        fixed_responses=dict(data.get("fixed_responses", {})),
    )


def _find_substring(text: str, patterns: tuple[str, ...]) -> str | None:
    lowered = text.lower()
    for pattern in patterns:
        if pattern.lower() in lowered:
            return pattern
    return None


def find_emergency(text: str, c: Constraints) -> str | None:
    return _find_substring(text, c.emergency_keywords)


def find_injection(text: str, c: Constraints) -> str | None:
    return _find_substring(text, c.injection_patterns)


def find_diagnosis(text: str, c: Constraints) -> str | None:
    return _find_substring(text, c.diagnosis_patterns)


def find_prescription(text: str, c: Constraints) -> str | None:
    return _find_substring(text, c.prescription_patterns)


def find_pii(text: str, c: Constraints) -> str | None:
    for pattern in c.pii_patterns:
        if re.search(pattern, text):
            return pattern
    return None


def scrub_pii(text: str, c: Constraints) -> str:
    scrubbed = text
    for pattern in c.pii_patterns:
        scrubbed = re.sub(pattern, "[thông tin cá nhân]", scrubbed)
    return scrubbed
