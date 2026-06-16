"""E2-2/E2-3/E3-1/E3-2 · anotación, propagación kNN, calidad y procedencia."""

import json

from acm.config import Taxonomy, load_profile_taxonomy, load_settings
from acm.models import AuditDecision, CandidateCard, CardScope
from acm.pipeline.classifier import classify, classify_fields
from acm.pipeline.normalizer import normalize_text
from acm.pipeline.propagation import (
    apply_high_confidence_propagation,
    propagate_facets_from_neighbors,
)
from acm.pipeline.normalizer import normalize_format
from acm.pipeline.quality import quality_flags, suggest_better_version, suggest_cloze
from acm.pipeline.similarity import DuplicateMatch, build_record_from_fields, make_fingerprint
from acm.store.registry import Registry


# --- E8-2 (contrato E3-1): flags de calidad ---

def test_quality_flag_empty_back():
    flags = quality_flags("What is X?", "")
    assert any(f["code"] == "back_vacio" for f in flags)


def test_quality_flag_long_answer():
    flags = quality_flags("Explain X", "palabra " * 80)
    assert any(f["code"] == "respuesta_larga" for f in flags)


def test_quality_flag_non_atomic():
    flags = quality_flags("List steps", "- uno\n- dos\n- tres")
    assert any(f["code"] == "no_atomica" for f in flags)


def test_quality_flag_recognition_yes_no():
    flags = quality_flags("¿Azure es de Microsoft?", "Sí")
    assert any(f["code"] == "reconocimiento" for f in flags)


def test_quality_flag_leak_shared_term():
    flags = quality_flags("What is the purpose of encryption?", "Encryption protects data")
    assert any(f["code"] == "pista_filtrada" for f in flags)


def test_quality_clean_card_no_flags():
    flags = quality_flags("What is a VPC?", "A logically isolated virtual network.")
    assert flags == []


# --- E8-1: detección de "mejor versión" ---

def test_better_version_when_new_has_fewer_flags():
    # vieja: respuesta sí/no (reconocimiento); nueva: recall real, limpia.
    result = suggest_better_version(
        new_front="What is a VPC?", new_back="A logically isolated virtual network in the cloud.",
        old_front="Is a VPC isolated?", old_back="Sí",
    )
    assert result["suggestion"] == "replace_old_with_new"


def test_better_version_keeps_existing_when_equal():
    result = suggest_better_version(
        new_front="What is a VPC?", new_back="A virtual network.",
        old_front="What is a VPC?", old_back="A virtual network.",
    )
    assert result["suggestion"] == "keep_existing"


def test_better_version_replace_when_more_complete():
    result = suggest_better_version(
        new_front="What is a VPC?",
        new_back="A logically isolated section of the cloud where you launch resources in a virtual network you define.",
        old_front="What is a VPC?", old_back="A network.",
    )
    assert result["suggestion"] == "replace_old_with_new"


# --- E8-3: sugerencia de cloze ---

def test_cloze_suggested_for_short_factual_answer():
    assert suggest_cloze("What protocol does HTTPS use for encryption?", "TLS") is True


def test_cloze_not_suggested_for_long_answer():
    assert suggest_cloze("Explain TLS", "A cryptographic protocol that secures communication over a network.") is False


def test_cloze_not_suggested_for_yes_no():
    assert suggest_cloze("Is HTTPS encrypted?", "Sí") is False


# --- E8-4: normalización de formato (preserva mayúsculas) ---

def test_normalize_format_strips_html_and_entities_keeps_case():
    assert normalize_format("<b>Azure</b>&nbsp;AD") == "Azure AD"


def test_normalize_format_collapses_whitespace():
    assert normalize_format("  What   is\n\nthis?  ") == "What is this?"


# --- E2-3: propagación de tags por kNN ---

def _match(cid, facets, score):
    rec = build_record_from_fields(
        candidate_id=cid, source="anki", origin_source="anki",
        front=cid, back="", scope=CardScope(facets=facets), note_type="Basic",
    )
    return DuplicateMatch(record=rec, score=score, reason_codes=("embedding_match",))


def test_propagation_votes_majority_facet():
    tax = Taxonomy(vendor=["aws", "azure"], topic=["compute", "storage"])
    matches = [
        _match("a", {"vendor": "aws", "topic": "compute"}, 0.90),
        _match("b", {"vendor": "aws"}, 0.85),
        _match("c", {"vendor": "azure"}, 0.80),
    ]
    result = propagate_facets_from_neighbors(matches, taxonomy=tax)
    assert result["vendor"]["value"] == "aws"  # 0.90+0.85 > 0.80
    assert result["vendor"]["votes"] == 2
    assert result["topic"]["value"] == "compute"


def test_propagation_ignores_invalid_taxonomy_values():
    tax = Taxonomy(vendor=["aws"])
    matches = [_match("a", {"vendor": "not-in-taxonomy"}, 0.95)]
    assert propagate_facets_from_neighbors(matches, taxonomy=tax) == {}


def test_apply_high_confidence_fills_missing_only():
    tax = Taxonomy(vendor=["aws"], topic=["compute"])
    classified = classify_fields(
        front="Q", back="A", source="x", suggested_tags=[], note_type="Basic",
        taxonomy=tax, fingerprint=make_fingerprint("Q", "A"),
    )
    applied = apply_high_confidence_propagation(
        classified, {"vendor": {"value": "aws", "confidence": 0.9, "votes": 3}}
    )
    assert "vendor::aws" in applied
    assert classified.scope.get("vendor") == "aws"

    # Baja confianza / pocos votos → no se auto-aplica.
    applied_low = apply_high_confidence_propagation(
        classified, {"topic": {"value": "compute", "confidence": 0.5, "votes": 1}}
    )
    assert applied_low == []
    assert classified.scope.get("topic") is None


# --- E3-2: procedencia (material_origen) ---

def test_material_origen_threads_and_persists(tmp_path):
    reg = Registry(tmp_path / "t.db")
    card = CandidateCard(front="Q", back="A", source="claude", material_origen="cap3.pdf#p12")
    fp = make_fingerprint(normalize_text("Q"), normalize_text("A"))
    classified = classify(card, Taxonomy(), fp)
    assert classified.material_origen == "cap3.pdf#p12"

    reg.insert(AuditDecision(card=classified, action="insert", reason="ok"))
    rows = reg.list_processed_cards()
    assert rows[0]["material_origen"] == "cap3.pdf#p12"


# --- E3-1: tool de anotación (sin persistir) ---

def test_acm_annotate_returns_annotations_without_persisting(tmp_path, monkeypatch):
    import acm.mcp_server as srv

    settings = load_settings()
    settings.acm.use_embeddings = False  # determinista, sin Ollama
    name, profile, taxonomy = load_profile_taxonomy(settings)
    reg = Registry(tmp_path / "t.db")

    def mock_setup(profile_name=None, *, include_registry=True):
        return reg, settings, name, profile, taxonomy

    monkeypatch.setattr(srv, "_setup", mock_setup)
    monkeypatch.setattr(srv, "_try_anki_client", lambda s: None)

    cards = [{
        "front": "What is Azure AD?", "back": "Cloud identity service",
        "source": "claude", "suggested_tags": ["vendor::azure", "topic::identity"],
        "material_origen": "az900.pdf",
    }]
    result = json.loads(srv.acm_annotate(json.dumps(cards)))

    assert result["summary"]["total"] == 1
    ann = result["annotations"][0]
    assert ann["es_duplicado"] is False
    assert "vendor::azure" in ann["tags_sugeridos"]
    assert ann["material_origen"] == "az900.pdf"
    assert ann["confianza"] in ("high", "medium", "low")
    assert "flags_calidad" in ann
    assert "sugerencia_cloze" in ann  # E8-3
    assert "formato_sugerido" in ann  # E8-4
    assert result["embeddings_used"] is False
    # E3-1: anotar NO persiste.
    assert reg.list_processed_cards() == []
