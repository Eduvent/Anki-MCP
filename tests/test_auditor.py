import pytest
from pathlib import Path

from acm.config import Settings, Taxonomy
from acm.models import CandidateCard
from acm.pipeline.auditor import audit_card, audit_batch
from acm.store.registry import Registry


def _taxonomy() -> Taxonomy:
    return Taxonomy(
        vendor=["azure", "aws"],
        topic=["identity", "networking"],
        cert=["az900"],
        type=["definition"],
    )


def _settings() -> Settings:
    return Settings()


@pytest.fixture
def registry(tmp_path):
    return Registry(tmp_path / "test.db")


def test_clean_card_gets_insert(registry):
    card = CandidateCard(
        front="What is Azure AD?",
        back="Cloud identity service",
        source="claude",
        suggested_tags=["vendor::azure", "topic::identity", "type::definition"],
    )
    decision = audit_card(card, registry, _taxonomy(), _settings())
    assert decision.action == "insert"


def test_duplicate_fingerprint_gets_possible_duplicate(registry):
    card = CandidateCard(
        front="What is Azure AD?",
        back="Cloud identity service",
        source="claude",
        suggested_tags=["vendor::azure", "topic::identity"],
    )
    # Primera ingesta
    d1 = audit_card(card, registry, _taxonomy(), _settings())
    registry.insert(d1)

    # Segunda ingesta idéntica
    d2 = audit_card(card, registry, _taxonomy(), _settings())
    assert d2.action == "possible_duplicate"


def test_semantic_duplicate_gets_possible_duplicate(registry):
    first = CandidateCard(
        front="Que es un CDN?",
        back="Content Delivery Network",
        source="claude",
        suggested_tags=["type::definition"],
    )
    second = CandidateCard(
        front="Dime la definicion de un CDN",
        back="Red de entrega de contenido",
        source="claude",
        suggested_tags=["type::definition"],
    )

    d1 = audit_card(first, registry, _taxonomy(), _settings())
    registry.insert(d1)

    d2 = audit_card(second, registry, _taxonomy(), _settings())
    assert d2.action == "possible_duplicate"
    assert "semant" in d2.reason.lower()


def test_too_many_unresolved_tags_gets_reject(registry):
    card = CandidateCard(
        front="What is XYZ?",
        back="Some service",
        source="claude",
        suggested_tags=[
            "vendor::invented1",
            "topic::invented2",
            "type::invented3",
        ],
    )
    decision = audit_card(card, registry, _taxonomy(), _settings())
    assert decision.action == "reject"
    assert "50%" in decision.reason


def test_no_tags_card_gets_insert(registry):
    card = CandidateCard(
        front="What is DNS?",
        back="Domain Name System",
        source="manual",
        suggested_tags=[],
    )
    decision = audit_card(card, registry, _taxonomy(), _settings())
    assert decision.action == "insert"


def test_batch_returns_all_decisions(registry):
    cards = [
        CandidateCard(front=f"Card {i}", back=f"Answer {i}", source="claude")
        for i in range(5)
    ]
    decisions = audit_batch(cards, registry, _taxonomy(), _settings())
    assert len(decisions) == 5
