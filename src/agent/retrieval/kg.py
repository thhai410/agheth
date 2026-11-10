"""Neo4j knowledge graph — Cypher đầy đủ (M1, architecture.md §6).

Adapter interface (architecture.md §4): mọi lời gọi Neo4j đi qua module này.
An toàn Cypher: predicate chỉ nhận giá trị trong whitelist (map sang rel type
cố định) — không bao giờ nội suy chuỗi từ LLM vào câu truy vấn.
"""
from __future__ import annotations

from dataclasses import dataclass

# canonical predicate (LLM extract) -> Neo4j relationship type (whitelist cố định)
PREDICATES: dict[str, str] = {
    "has_symptom": "HAS_SYMPTOM",
    "treats": "TREATS",
    "indicates": "INDICATES",
    "contraindicated": "CONTRAINDICATED",
    "associated_with": "ASSOCIATED_WITH",
}
_REVERSE_PREDICATES = {rel: pred for pred, rel in PREDICATES.items()}


@dataclass(frozen=True)
class Triplet:
    subject: str
    predicate: str
    object: str
    source_row_id: str = ""


def validate_triplet(triplet: Triplet) -> str | None:
    """Trả về lý do lỗi, hoặc None nếu hợp lệ (fail-closed)."""
    subject = triplet.subject.strip()
    obj = triplet.object.strip()
    if not subject or not obj:
        return "thiếu subject/object"
    if len(subject) > 200 or len(obj) > 200:
        return "thuật ngữ quá dài (>200 ký tự)"
    if triplet.predicate not in PREDICATES:
        return f"predicate không có trong whitelist: {triplet.predicate!r}"
    if subject.lower() == obj.lower():
        return "subject trùng object"
    return None


def normalize_name(name: str) -> str:
    return " ".join(name.split()).strip()


class Neo4jKgStore:
    """Ingest + truy vấn triplet trên Neo4j."""

    def __init__(self, uri: str, user: str, password: str):
        from neo4j import GraphDatabase  # lazy

        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        self._driver.close()

    def ensure_schema(self) -> None:
        with self._driver.session() as session:
            session.run("CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)")

    def ingest(self, triplets: list[Triplet]) -> dict:
        """Validate + dedupe + MERGE. Trả về report {valid, invalid, skipped_dup}."""
        report = {"valid": 0, "invalid": 0, "skipped_dup": 0}
        seen: set[tuple[str, str, str]] = set()
        with self._driver.session() as session:
            for triplet in triplets:
                error = validate_triplet(triplet)
                if error:
                    report["invalid"] += 1
                    continue
                key = (
                    normalize_name(triplet.subject).lower(),
                    triplet.predicate,
                    normalize_name(triplet.object).lower(),
                )
                if key in seen:
                    report["skipped_dup"] += 1
                    continue
                seen.add(key)
                session.run(self._merge_cypher(triplet.predicate), {
                    "subject": normalize_name(triplet.subject),
                    "object": normalize_name(triplet.object),
                    "source_id": triplet.source_row_id,
                })
                report["valid"] += 1
        return report

    @staticmethod
    def _merge_cypher(predicate: str) -> str:
        rel = PREDICATES[predicate]  # whitelist — an toàn nội suy vào Cypher
        return (
            "MERGE (s:Entity {name: $subject}) "
            "MERGE (o:Entity {name: $object}) "
            f"MERGE (s)-[r:{rel}]->(o) "
            "SET r.sources = CASE "
            "  WHEN $source_id = '' THEN coalesce(r.sources, []) "
            "  WHEN $source_id IN coalesce(r.sources, []) THEN r.sources "
            "  ELSE coalesce(r.sources, []) + $source_id END"
        )

    def query(self, entities: list[str], limit: int = 20) -> list[Triplet]:
        """Trả về triplet liên quan tới entities (match 2 chiều, case-insensitive)."""
        lowered = [normalize_name(e).lower() for e in entities if e.strip()]
        if not lowered:
            return []
        cypher = (
            "MATCH (s:Entity)-[r]->(o:Entity) "
            "WHERE toLower(s.name) IN $entities OR toLower(o.name) IN $entities "
            "RETURN s.name AS subject, type(r) AS rel, o.name AS object "
            "LIMIT $limit"
        )
        with self._driver.session() as session:
            result = session.run(cypher, {"entities": lowered, "limit": limit})
            return [
                Triplet(
                    subject=record["subject"],
                    predicate=_REVERSE_PREDICATES.get(record["rel"], "associated_with"),
                    object=record["object"],
                )
                for record in result
            ]
