"""2 tool mặc định (architecture.md §8) — thuần dữ liệu, không phải LLM đoán.

- drug_lookup:     tra bảng thuốc nhỏ configs/drugs.json
- symptom_checker: triệu chứng -> bệnh liên quan qua Neo4j KG
"""
from __future__ import annotations

import json
import unicodedata

from agent.safety.constraints import config_dir

DISCLAIMER = "Thông tin demo tham khảo — không thay thế tư vấn y khoa."


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower().strip()
    return " ".join(text.split())


def drug_lookup(name: str) -> dict:
    path = config_dir() / "drugs.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    drugs = data.get("drugs", {})
    target = _norm(name)
    for drug_name, info in drugs.items():
        aliases = [_norm(alias) for alias in info.get("aliases", [])]
        if target in [_norm(drug_name), *aliases]:
            return {
                "found": True,
                "name": drug_name,
                "class": info.get("class", ""),
                "common_uses": info.get("common_uses", []),
                "notes": info.get("notes", ""),
                "disclaimer": DISCLAIMER,
            }
    return {"found": False, "name": name, "disclaimer": DISCLAIMER}


def symptom_checker(symptoms: list[str], kg=None) -> dict:
    """Triệu chứng -> bệnh liên quan qua KG (predicate indicates/has_symptom)."""
    if kg is None:
        from agent.config import get_settings
        from agent.retrieval.kg import Neo4jKgStore

        settings = get_settings()
        kg = Neo4jKgStore(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password)
    triplets = kg.query([_norm(s) for s in symptoms])
    # bệnh = subject của (bệnh)-[has_symptom]->(triệu chứng)
    #        hoặc object của (triệu chứng)-[indicates]->(bệnh)
    related = sorted({
        t.subject for t in triplets if t.predicate == "has_symptom"
    } | {
        t.object for t in triplets if t.predicate == "indicates"
    })
    return {
        "symptoms": symptoms,
        "related_conditions": related,
        "evidence": [
            {"subject": t.subject, "predicate": t.predicate, "object": t.object}
            for t in triplets
        ],
        "disclaimer": DISCLAIMER,
    }


def default_specs() -> list:
    from agent.tools.registry import ToolSpec

    return [
        ToolSpec(
            name="drug_lookup",
            description="Tra thông tin thuốc theo tên (tên generic hoặc biệt dược).",
            params={"name": "str"},
        ),
        ToolSpec(
            name="symptom_checker",
            description="Liệt kê bệnh có thể liên quan tới các triệu chứng (qua knowledge graph).",
            params={"symptoms": "list"},
        ),
    ]


def default_registry(executor=None):
    from agent.tools.registry import ToolRegistry

    return ToolRegistry(default_specs(), executor=executor)
