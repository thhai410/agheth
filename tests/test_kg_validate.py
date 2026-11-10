"""Unit tests cho validate_triplet + whitelist predicate (fail-closed)."""

from agent.retrieval.kg import PREDICATES, Triplet, validate_triplet


class TestValidateTriplet:
    def test_valid(self):
        t = Triplet(subject="sốt", predicate="has_symptom", object="cảm cúm")
        assert validate_triplet(t) is None

    def test_all_whitelisted_predicates_valid(self):
        for predicate in PREDICATES:
            t = Triplet(subject="A", predicate=predicate, object="B")
            assert validate_triplet(t) is None

    def test_rejects_unknown_predicate(self):
        t = Triplet(subject="A", predicate="causes_always", object="B")
        error = validate_triplet(t)
        assert error and "whitelist" in error

    def test_rejects_empty_terms(self):
        assert validate_triplet(Triplet("", "treats", "B")) is not None
        assert validate_triplet(Triplet("A", "treats", "   ")) is not None

    def test_rejects_same_subject_object(self):
        t = Triplet(subject="Sốt", predicate="indicates", object="sốt")
        error = validate_triplet(t)
        assert error and "trùng" in error

    def test_rejects_too_long_terms(self):
        t = Triplet(subject="x" * 201, predicate="treats", object="B")
        assert validate_triplet(t) is not None
