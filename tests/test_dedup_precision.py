"""Regresiones del reporte 2026-06-16 — precisión de deduplicación.

Reproduce los falsos positivos reales (cross-vendor / embedding-solo / intra-lote)
y verifica que el motor ya NO los marca como duplicados, sin perder los dups reales.
"""

import dataclasses
import math

import pytest

from acm.config import Settings, Taxonomy
from acm.models import CandidateCard, CardScope
from acm.pipeline.auditor import audit_batch
from acm.pipeline.similarity import build_record_from_fields, compare_records
from acm.store.registry import Registry


def _unit(cos: float) -> list[float]:
    """Vector unitario en 2D con coseno `cos` respecto de [1, 0]."""
    return [cos, math.sqrt(max(0.0, 1.0 - cos * cos))]


def _rec(cid, front, back, vendor=None, *, source="anki", cos=None):
    rec = build_record_from_fields(
        candidate_id=cid, source=source, origin_source=source,
        front=front, back=back,
        scope=CardScope(facets={"vendor": vendor} if vendor else {}),
        note_type="Básico",
    )
    if cos is not None:
        rec = dataclasses.replace(rec, features=dataclasses.replace(rec.features, embedding=tuple(cos)))
    return rec


# --- §3: el falso positivo central ---

def test_cross_vendor_embedding_only_is_not_a_match():
    """Azure(es) vs AWS(en), coseno ~0.62, sin solape léxico → NO es duplicado."""
    q = _rec("q", "¿Qué es Azure Cloud Shell?", "Un shell en el navegador", "azure",
             source="input", cos=_unit(1.0))
    m = _rec("m", "What is the AWS CLI?", "A tool to run commands", "aws",
             cos=_unit(0.62))
    edge = compare_records(q, m, similar_threshold=0.75)
    assert edge is None  # antes: possible_duplicate con score ~0.90


def test_cross_vendor_high_cosine_still_not_a_match():
    """Aún con coseno alto, distinto vendor sin léxico no alcanza para duplicado."""
    q = _rec("q", "Cloud Shell usa cifrado doble en reposo", "...", "azure", source="input", cos=_unit(1.0))
    m = _rec("m", "Vault dev TLS flag", "-dev-tls", "hashicorp", cos=_unit(0.93))
    edge = compare_records(q, m, similar_threshold=0.75)
    assert edge is None


def test_same_vendor_reformulation_still_matches():
    """Recall preservado: misma pregunta reformulada, mismo vendor → sí matchea."""
    q = _rec("q", "¿Dónde almacena archivos Azure Cloud Shell?", "En un Azure File Share", "azure",
             source="input", cos=_unit(1.0))
    m = _rec("m", "Azure Cloud Shell almacena archivos dónde", "Usa un File Share de Azure", "azure",
             cos=_unit(0.9))
    edge = compare_records(q, m, similar_threshold=0.75)
    assert edge is not None  # comparten tokens (corroborado) + mismo vendor


def test_exact_duplicate_still_flagged_regardless_of_vendor():
    """Un duplicado EXACTO sigue siendo duplicado (idempotencia depende de esto)."""
    q = _rec("q", "¿Qué es Azure Cloud Shell?", "Un shell en el navegador", "azure", source="input", cos=_unit(1.0))
    m = _rec("m", "¿Qué es Azure Cloud Shell?", "Un shell en el navegador", "azure", cos=_unit(0.5))
    edge = compare_records(q, m, similar_threshold=0.75)
    assert edge is not None
    assert edge.score == pytest.approx(1.0)
    assert "exact_fingerprint" in edge.reason_codes


# --- §8: dedup intra-lote ---

def _tax():
    return Taxonomy(vendor=["azure"], topic=["compute"], type=["definition"])


def test_intra_batch_siblings_not_escalated(tmp_path):
    settings = Settings()
    settings.acm.use_embeddings = False  # determinista; fuerza la vía léxica
    reg = Registry(tmp_path / "t.db")
    cards = [
        CandidateCard(front="What is Azure Cloud Shell?", back="A browser-based shell", source="claude"),
        CandidateCard(front="What is Azure Cloud Shell exactly?", back="A shell in the browser", source="claude"),
        CandidateCard(front="What two shell experiences does Azure Cloud Shell offer?",
                      back="Bash and PowerShell", source="claude"),
    ]
    decisions = audit_batch(cards, reg, _tax(), settings)
    # Un lote temático redactado a propósito NO debe auto-marcarse como duplicado.
    assert all(d.action != "possible_duplicate" for d in decisions)


def test_real_registry_duplicate_still_flagged(tmp_path):
    from acm.models import AuditDecision
    settings = Settings()
    settings.acm.use_embeddings = False
    reg = Registry(tmp_path / "t.db")
    card = CandidateCard(front="What is Azure Cloud Shell?", back="A browser-based shell", source="claude")
    first = audit_batch([card], reg, _tax(), settings)[0]
    reg.insert(first)
    # Re-auditar la misma card → duplicado contra el registro (no intra-lote).
    second = audit_batch([card], reg, _tax(), settings)[0]
    assert second.action == "possible_duplicate"
    assert any("exact_fingerprint" in m.get("reason_codes", []) for m in second.match_details)
