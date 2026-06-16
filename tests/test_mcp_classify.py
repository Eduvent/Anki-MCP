"""Tests para las herramientas MCP de auto-clasificación token-eficiente."""

import json

from acm.config import load_profile_taxonomy, load_settings
from acm.mcp_server import acm_auto_classify, acm_apply_tags, acm_list_untagged, _classify_confidence
from acm.pipeline.classifier import classify_fields
from acm.pipeline.normalizer import normalize_text
from acm.pipeline.similarity import make_fingerprint


def test_auto_classify_high_confidence():
    """Cards con vendor+topic+tags resueltos van a classified."""
    cards = [
        {
            "front": "What is Azure AD?",
            "back": "Identity service",
            "source": "claude",
            "suggested_tags": ["vendor::azure", "topic::identity", "type::definition"],
        }
    ]
    result = json.loads(acm_auto_classify(json.dumps(cards)))

    assert result["summary"]["auto_classified"] == 1
    assert result["summary"]["needs_review"] == 0
    assert result["classified"][0]["vendor"] == "azure"
    assert result["classified"][0]["confidence"] == "high"
    assert "taxonomy" not in result  # No incluye taxonomía si no hay review


def test_auto_classify_low_confidence_goes_to_review():
    """Cards sin vendor ni tags van a needs_review."""
    cards = [
        {
            "front": "What is DNS?",
            "back": "Domain Name System",
        }
    ]
    result = json.loads(acm_auto_classify(json.dumps(cards)))

    assert result["summary"]["auto_classified"] == 0
    assert result["summary"]["needs_review"] == 1
    assert result["needs_review"][0]["confidence"] in ("low", "medium")
    assert "taxonomy" in result  # Incluye taxonomía para que el agente decida


def test_auto_classify_mixed_batch():
    """Batch con tarjetas de alta y baja confianza se separa correctamente."""
    cards = [
        {
            "front": "What is Azure AD?",
            "back": "Identity service",
            "suggested_tags": ["vendor::azure", "topic::identity", "type::definition"],
        },
        {
            "front": "How to configure a firewall?",
            "back": "Steps to configure",
        },
        {
            "front": "What is EC2?",
            "back": "Virtual machine in AWS",
            "suggested_tags": ["vendor::aws", "topic::compute", "type::definition"],
        },
    ]
    result = json.loads(acm_auto_classify(json.dumps(cards)))

    assert result["summary"]["total"] == 3
    assert result["summary"]["auto_classified"] == 2
    assert result["summary"]["needs_review"] == 1
    assert result["summary"]["tokens_saved_pct"] == 67


def test_auto_classify_uses_deck_regex_and_keywords():
    cards = [
        {
            "front": "What does IAM stand for?",
            "back": "Identity and Access Management",
            "deck": "Cloud Certs::AWS::AWS Cloud Practitioner",
        }
    ]
    result = json.loads(acm_auto_classify(json.dumps(cards)))

    assert result["summary"]["auto_classified"] == 1
    classified = result["classified"][0]
    assert classified["vendor"] == "aws"
    assert classified["cert"] == "clf-c02"
    assert classified["topic"] == "identity"
    assert classified["type"] == "acronym"
    assert "vendor::aws" in classified["tags_resolved"]
    assert "cert::clf-c02" in classified["tags_resolved"]
    assert "topic::identity" in classified["tags_resolved"]
    assert "type::acronym" in classified["tags_resolved"]


def test_auto_classify_includes_hints_for_ambiguous():
    """Tarjetas ambiguas incluyen hints sobre qué falta."""
    cards = [
        {
            "front": "What is Azure pricing?",
            "back": "Pricing overview",
        }
    ]
    result = json.loads(acm_auto_classify(json.dumps(cards)))

    # Azure se detecta por keyword, pero el topic sigue ambiguo
    review = result["needs_review"]
    assert len(review) == 1
    assert review[0]["hints"]["detected_vendor"] == "azure"
    assert "topic" in review[0]["hints"]["missing"]


def test_auto_classify_invalid_input():
    result = json.loads(acm_auto_classify(json.dumps("not a list")))
    assert "error" in result


def test_apply_tags_new_card(tmp_path, monkeypatch):
    """acm_apply_tags con card nueva la procesa via ingest pipeline."""
    # Monkeypatch _setup para usar un registry temporal
    from acm.config import load_profile_taxonomy, load_settings
    from acm.store.registry import Registry

    settings = load_settings()
    name, profile, taxonomy = load_profile_taxonomy(settings)
    test_registry = Registry(tmp_path / "test.db")

    def mock_setup(profile_name=None, *, include_registry=True):
        return test_registry, settings, name, profile, taxonomy

    monkeypatch.setattr("acm.mcp_server._setup", mock_setup)

    assignments = [
        {
            "card": {
                "front": "What is Azure AD?",
                "back": "Identity service",
                "source": "claude",
            },
            "tags": ["vendor::azure", "topic::identity", "type::definition"],
        }
    ]
    result = json.loads(acm_apply_tags(json.dumps(assignments)))

    assert result["summary"]["cards_ingested"] == 1
    assert result["summary"]["errors"] == 0
    assert result["cards_ingested"][0]["vendor"] == "azure"


def test_apply_tags_accepts_file_input(tmp_path, monkeypatch):
    from acm.config import load_profile_taxonomy
    from acm.store.registry import Registry

    settings = load_settings()
    name, profile, taxonomy = load_profile_taxonomy(settings)
    test_registry = Registry(tmp_path / "test-file.db")

    def mock_setup(profile_name=None, *, include_registry=True):
        return test_registry, settings, name, profile, taxonomy

    monkeypatch.setattr("acm.mcp_server._setup", mock_setup)

    assignments = [
        {
            "card": {
                "front": "What is Azure AD?",
                "back": "Identity service",
                "source": "claude",
            },
            "tags": ["vendor::azure", "topic::identity", "type::definition"],
        }
    ]
    assignments_path = tmp_path / "assignments.json"
    assignments_path.write_text(json.dumps(assignments), encoding="utf-8")

    result = json.loads(acm_apply_tags(file=str(assignments_path)))

    assert result["summary"]["cards_ingested"] == 1
    assert result["summary"]["errors"] == 0
    assert result["cards_ingested"][0]["vendor"] == "azure"


def _make_fake_anki_client(settings, deck_name="Cloud Certs::AWS::AWS Cloud Practitioner", *, include_deck_in_notes=True):
    """Factory para crear FakeAnkiClient con o sin deckName en notesInfo."""

    class FakeAnkiClient:
        instances = []

        def __init__(self, _url):
            self.added = []
            base_notes = [
                {
                    "noteId": 1,
                    "tags": [],
                    "modelName": "Basic",
                    "fields": {
                        settings.anki.field_mapping.front: {"value": "What does IAM stand for?"},
                        settings.anki.field_mapping.back: {"value": "Identity and Access Management"},
                    },
                },
                {
                    "noteId": 2,
                    "tags": [],
                    "modelName": "Basic",
                    "fields": {
                        settings.anki.field_mapping.front: {"value": "Explain cloud elasticity."},
                        settings.anki.field_mapping.back: {"value": "Scaling concept"},
                    },
                },
                {
                    "noteId": 3,
                    "tags": [],
                    "modelName": "Basic",
                    "fields": {
                        settings.anki.field_mapping.front: {"value": "What is a pricing model?"},
                        settings.anki.field_mapping.back: {"value": "A cost concept"},
                    },
                },
            ]
            if include_deck_in_notes:
                for note in base_notes:
                    note["deckName"] = deck_name
            self.notes = base_notes
            self._deck_name = deck_name
            FakeAnkiClient.instances.append(self)

        def is_available(self):
            return True

        def expand_decks(self, deck, include_subdecks=True):
            return [self._deck_name]

        def find_notes(self, query):
            return [note["noteId"] for note in self.notes]

        def get_notes_info(self, note_ids):
            return [note for note in self.notes if note["noteId"] in note_ids]

        def add_tags(self, note_ids, tags):
            self.added.append({"note_ids": note_ids, "tags": set(tags.split())})

        def close(self):
            return None

    return FakeAnkiClient


def test_list_untagged_auto_applies_and_paginates(monkeypatch):
    settings = load_settings()
    resolved_profile_name, profile, taxonomy = load_profile_taxonomy(settings)

    FakeAnkiClient = _make_fake_anki_client(settings)

    def mock_setup(profile_name=None, *, include_registry=True):
        return None, settings, resolved_profile_name, profile, taxonomy

    monkeypatch.setattr("acm.mcp_server.AnkiConnectClient", FakeAnkiClient)
    monkeypatch.setattr("acm.mcp_server._setup", mock_setup)

    result = json.loads(acm_list_untagged("Cloud Certs", limit=1, offset=0))

    assert result["summary"]["total_scanned"] == 3
    assert result["summary"]["auto_tagged"] == 1
    assert result["summary"]["needs_review"] == 2
    assert "auto_tagged" not in result
    assert result["pagination"]["returned"] == 1
    assert result["pagination"]["total"] == 2
    assert result["pagination"]["has_more"] is True
    assert result["pagination"]["next_offset"] == 1

    review_item = result["needs_review"][0]
    assert review_item["note_id"] == 2
    assert review_item["front"] == "Explain cloud elasticity."
    assert review_item["missing"] == ["topic"]
    assert review_item["deck"] == "Cloud Certs::AWS::AWS Cloud Practitioner"
    assert "hints" in review_item
    # Deck rules resolved vendor+cert even for ambiguous cards
    assert review_item["hints"].get("vendor") == "aws"
    assert review_item["hints"].get("cert") == "clf-c02"
    assert "taxonomy" in result

    added = FakeAnkiClient.instances[0].added
    assert len(added) == 1
    assert added[0]["note_ids"] == [1]
    assert {"vendor::aws", "cert::clf-c02", "topic::identity", "type::acronym"} <= added[0]["tags"]


def test_list_untagged_deck_mapping_without_deckname_in_notes(monkeypatch):
    """Verifica que el mapeo note→deck funciona incluso sin deckName en notesInfo.

    AnkiConnect real no incluye deckName en notesInfo; el mapeo se construye
    desde la iteración por sub-decks en expand_decks + find_notes.
    """
    settings = load_settings()
    resolved_profile_name, profile, taxonomy = load_profile_taxonomy(settings)

    # include_deck_in_notes=False simula el comportamiento real de AnkiConnect
    FakeAnkiClient = _make_fake_anki_client(settings, include_deck_in_notes=False)

    def mock_setup(profile_name=None, *, include_registry=True):
        return None, settings, resolved_profile_name, profile, taxonomy

    monkeypatch.setattr("acm.mcp_server.AnkiConnectClient", FakeAnkiClient)
    monkeypatch.setattr("acm.mcp_server._setup", mock_setup)

    result = json.loads(acm_list_untagged("Cloud Certs"))

    # Sin el fix, auto_tagged sería 0 porque deck="Cloud Certs" no matchea
    # Con el fix, note_to_deck mapea al sub-deck real → deck_tag_rules resuelven vendor+cert
    assert result["summary"]["auto_tagged"] == 1
    assert result["summary"]["needs_review"] == 2

    # Verify the ambiguous cards still got deck + partial hints
    for item in result["needs_review"]:
        assert item["deck"] == "Cloud Certs::AWS::AWS Cloud Practitioner"
        assert item["hints"]["vendor"] == "aws"
        assert item["hints"]["cert"] == "clf-c02"


def test_apply_tags_invalid_format():
    """Tags sin formato category::value son rechazados."""
    assignments = [
        {
            "card": {"front": "Test", "back": "Test", "source": "manual"},
            "tags": ["bad-tag-format"],
        }
    ]
    result = json.loads(acm_apply_tags(json.dumps(assignments)))

    assert result["summary"]["errors"] == 1
    assert "formato inválido" in result["errors"][0]["error"]


def test_apply_tags_missing_target():
    """Asignación sin note_id ni card genera error."""
    assignments = [{"tags": ["vendor::azure"]}]
    result = json.loads(acm_apply_tags(json.dumps(assignments)))

    assert result["summary"]["errors"] == 1
    assert "note_id" in result["errors"][0]["error"]


def test_classify_confidence_levels():
    """Verifica los niveles de confianza del clasificador."""
    from acm.config import load_profile_taxonomy, load_settings

    settings = load_settings()
    # E2-1: la detección por keyword es data-driven; el perfil default trae las
    # reglas de dominio desde settings.yaml.
    _, profile, taxonomy = load_profile_taxonomy(settings)

    # High: vendor + topic + 2+ resolved + no unresolved
    high = classify_fields(
        front="What is Azure AD?", back="Identity service", source="claude",
        suggested_tags=["vendor::azure", "topic::identity"],
        note_type="Basic", taxonomy=taxonomy,
        fingerprint=make_fingerprint("x", "y"), profile=profile,
    )
    assert _classify_confidence(high) == "high"

    # Medium: vendor detected (keyword) but missing topic tag
    medium = classify_fields(
        front="Azure basics", back="Overview", source="claude",
        suggested_tags=[], note_type="Basic", taxonomy=taxonomy,
        fingerprint=make_fingerprint("x", "y"), profile=profile,
    )
    assert _classify_confidence(medium) == "medium"

    # Low: nothing detected
    low = classify_fields(
        front="How to cook pasta?", back="Boil water", source="manual",
        suggested_tags=[], note_type="Basic", taxonomy=taxonomy,
        fingerprint=make_fingerprint("x", "y"), profile=profile,
    )
    assert _classify_confidence(low) == "low"
