"""Tests de deduplicación — opera directamente sobre el motor de similarity."""

import pytest

from acm.config import Taxonomy
from acm.models import AuditDecision, CandidateCard
from acm.pipeline.classifier import classify
from acm.pipeline.normalizer import normalize_text
from acm.pipeline.similarity import (
    build_record_from_classified,
    build_record_from_registry_row,
    find_similar_records,
    make_fingerprint,
)
from acm.store.registry import Registry


def _taxonomy() -> Taxonomy:
    return Taxonomy(
        vendor=["azure", "aws"],
        topic=["identity", "networking"],
        cert=[],
        type=["definition"],
    )


def _make_card(front: str, back: str, source: str = "claude") -> CandidateCard:
    return CandidateCard(front=front, back=back, source=source)


def _classified(card: CandidateCard, taxonomy: Taxonomy):
    fp = make_fingerprint(normalize_text(card.front), normalize_text(card.back))
    return classify(card, taxonomy, fp)


def _check_duplicates(classified, registry):
    """Busca duplicados fuertes (score >= 0.90) en el registry."""
    query = build_record_from_classified(
        classified, candidate_id="query", source="input", origin_source=classified.source,
    )
    candidates = [
        build_record_from_registry_row(row)
        for row in registry.list_processed_cards()
    ]
    matches = find_similar_records(query, candidates, similar_threshold=0.75)
    strong = [m for m in matches if m.score >= 0.90]
    return strong


@pytest.fixture
def registry(tmp_path):
    return Registry(tmp_path / "test.db")


def test_no_duplicate_on_empty_registry(registry):
    taxonomy = _taxonomy()
    card = _make_card("What is Azure AD?", "Identity service")
    classified = _classified(card, taxonomy)
    matches = _check_duplicates(classified, registry)
    assert not matches


def test_fingerprint_duplicate_detected(registry):
    taxonomy = _taxonomy()
    card = _make_card("What is Azure AD?", "Identity service")
    classified = _classified(card, taxonomy)

    decision = AuditDecision(card=classified, action="insert", reason="ok")
    registry.insert(decision)

    matches = _check_duplicates(classified, registry)
    assert matches
    assert any("exact_fingerprint" in code for m in matches for code in m.reason_codes)


def test_front_same_scope_duplicate(registry):
    taxonomy = _taxonomy()
    card1 = _make_card("What is Azure AD?", "Identity service v1")
    card2 = _make_card("What is Azure AD?", "Identity service v2")

    classified1 = _classified(card1, taxonomy)
    classified2 = _classified(card2, taxonomy)

    assert classified1.fingerprint != classified2.fingerprint

    decision1 = AuditDecision(card=classified1, action="insert", reason="ok")
    registry.insert(decision1)

    matches = _check_duplicates(classified2, registry)
    assert matches


def test_anki_front_duplicate(registry):
    taxonomy = _taxonomy()
    card = _make_card("What is EC2?", "Virtual machine")
    classified = _classified(card, taxonomy)

    query = build_record_from_classified(
        classified, candidate_id="query", source="input", origin_source="claude",
    )
    from acm.models import CardScope
    from acm.pipeline.similarity import build_record_from_fields
    anki_record = build_record_from_fields(
        candidate_id="anki:12345",
        source="anki",
        origin_source="anki",
        front="What is EC2?",
        back="Virtual machine service",
        scope=CardScope(),
        note_type="Basic",
        anki_note_id=12345,
    )
    matches = find_similar_records(query, [anki_record], similar_threshold=0.75)
    assert matches
    assert "anki:12345" == matches[0].record.candidate_id


def test_semantic_definition_duplicate_detected(registry):
    taxonomy = _taxonomy()
    card1 = _make_card("Que es un CDN?", "Content Delivery Network")
    card2 = _make_card("Dime la definicion de un CDN", "Red de entrega de contenido")

    classified1 = _classified(card1, taxonomy)
    classified2 = _classified(card2, taxonomy)

    decision1 = AuditDecision(card=classified1, action="insert", reason="ok")
    registry.insert(decision1)

    matches = _check_duplicates(classified2, registry)
    assert matches
    assert any("semantic_key_match" in code for m in matches for code in m.reason_codes)


def test_make_fingerprint_deterministic():
    fp1 = make_fingerprint("azure ad", "identity service")
    fp2 = make_fingerprint("azure ad", "identity service")
    assert fp1 == fp2


def test_make_fingerprint_different_inputs():
    fp1 = make_fingerprint("azure ad", "identity service")
    fp2 = make_fingerprint("azure ad", "access management service")
    assert fp1 != fp2
