"""E7-2 / E7-3 / C-3 · mantenimiento, reorganización dry-run y contratos MCP."""

import json

from acm.config import load_profile_taxonomy, load_settings
from acm.models import AuditDecision, CardScope, ClassifiedCard
from acm.pipeline.duplicate_audit import collection_health
from acm.pipeline.normalizer import normalize_text
from acm.pipeline.similarity import make_fingerprint
from acm.store.registry import Registry


def _card(front, back):
    return ClassifiedCard(
        front=front, back=back, front_normalized=normalize_text(front),
        back_normalized=normalize_text(back), scope=CardScope(),
        tags_resolved=[], tags_unresolved=[], note_type="Basic",
        fingerprint=make_fingerprint(normalize_text(front), normalize_text(back)), source="claude",
    )


def _settings_no_embed():
    settings = load_settings()
    settings.acm.use_embeddings = False
    return settings


def test_collection_health_counts_untagged(tmp_path):
    settings = _settings_no_embed()
    name, profile, taxonomy = load_profile_taxonomy(settings)
    reg = Registry(tmp_path / "t.db")
    for front in ("alpha", "beta", "gamma"):
        reg.insert(AuditDecision(card=_card(front, front + " answer"), action="insert", reason="x"))

    report = collection_health(
        registry=reg, settings=settings, taxonomy=taxonomy,
        profile_name=name, profile=profile, deck=None, include_subdecks=True, anki_client=None,
    )
    assert report["cards_scanned"] == 3
    assert report["untagged"] == 3  # E7-2: huérfanas sin tag
    assert report["leeches"] == 0   # sin Anki


def test_acm_stats_contract(tmp_path, monkeypatch):
    """C-3: contrato de salida de acm_stats."""
    import acm.mcp_server as srv
    reg = Registry(tmp_path / "t.db")
    reg.insert(AuditDecision(card=_card("q", "a"), action="insert", reason="ok"))
    monkeypatch.setattr(srv, "_setup", lambda *a, **k: (reg, load_settings(), "default", None, None))

    result = json.loads(srv.acm_stats())
    assert "by_action" in result
    assert "metrics" in result
    assert "sync_batches" in result
    assert result["metrics"]["auto_resolved"] == 1  # E9-4


def test_acm_reorganize_dry_run_does_not_mutate(tmp_path, monkeypatch):
    """E7-3 + E9-1: dry-run previsualiza sin tocar nada."""
    import acm.mcp_server as srv
    settings = _settings_no_embed()
    name, profile, taxonomy = load_profile_taxonomy(settings)
    reg = Registry(tmp_path / "t.db")
    monkeypatch.setattr(srv, "_setup", lambda *a, **k: (reg, settings, name, profile, taxonomy))
    monkeypatch.setattr(srv, "_try_anki_client", lambda s: None)

    result = json.loads(srv.acm_reorganize("Cloud Certs", dry_run=True))
    assert result["dry_run"] is True
    assert "plan" in result


def test_acm_audit_maintenance_mode(tmp_path, monkeypatch):
    """E7-2: acm_audit(mode=maintenance) devuelve el reporte de salud."""
    import acm.mcp_server as srv
    settings = _settings_no_embed()
    name, profile, taxonomy = load_profile_taxonomy(settings)
    reg = Registry(tmp_path / "t.db")
    monkeypatch.setattr(srv, "_setup", lambda *a, **k: (reg, settings, name, profile, taxonomy))
    monkeypatch.setattr(srv, "_try_anki_client", lambda s: None)

    result = json.loads(srv.acm_audit("Cloud Certs", mode="maintenance"))
    assert "duplicate_clusters" in result
    assert "untagged" in result
    assert "leeches" in result
