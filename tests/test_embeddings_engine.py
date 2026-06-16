"""E1 · Tests deterministas del motor de embeddings (Ollama mockeado).

Cubre el camino más riesgoso y menos determinista (review §13): cache por
fingerprint, calibración coseno, recuperación kNN y degradación si Ollama cae.
"""

import dataclasses
import math

import pytest

from acm.models import CardScope
from acm.pipeline.similarity import (
    audit_duplicate_records,
    build_record_from_fields,
    calibrate_cosine,
    cosine_floor_for,
    embedding_text,
    enrich_records_with_embeddings,
)
from acm.store.registry import Registry, embedding_cache_io


def _rec(cid: str, front: str, back: str = ""):
    return build_record_from_fields(
        candidate_id=cid, source="registry", origin_source="manual",
        front=front, back=back, scope=CardScope(), note_type="Basic",
    )


def _emb(rec, vec):
    return dataclasses.replace(rec, features=dataclasses.replace(rec.features, embedding=tuple(vec)))


# --- E1-6 calibración ---

def test_calibrate_cosine_anchors():
    assert calibrate_cosine(0.20) == 0.0
    assert calibrate_cosine(0.10) == 0.0
    assert calibrate_cosine(0.40) == pytest.approx(0.75)
    assert calibrate_cosine(0.55) == pytest.approx(0.90)
    assert calibrate_cosine(1.0) == 1.0


def test_calibrate_cosine_monotonic():
    prev = -1.0
    c = 0.0
    while c <= 1.0001:
        v = calibrate_cosine(c)
        assert v >= prev - 1e-9
        prev = v
        c += 0.02


def test_cosine_floor_is_inverse_of_calibrate():
    assert cosine_floor_for(0.75) == pytest.approx(0.40)
    assert cosine_floor_for(0.90) == pytest.approx(0.55)
    for raw in (0.30, 0.45, 0.62):
        assert cosine_floor_for(calibrate_cosine(raw)) == pytest.approx(raw, abs=1e-6)


# --- E1-3 cache ---

def test_embedding_cache_roundtrip_and_model_scoped(tmp_path):
    reg = Registry(tmp_path / "t.db")
    reg.put_cached_embeddings({"fp1": [0.1, 0.2, 0.3], "fp2": [1.0, 0.0]}, "m1")
    got = reg.get_cached_embeddings(["fp1", "fp2", "missing"], "m1")
    assert got["fp1"] == [0.1, 0.2, 0.3]
    assert got["fp2"] == [1.0, 0.0]
    assert "missing" not in got
    # keyed por modelo → otro modelo no ve los vectores
    assert reg.get_cached_embeddings(["fp1"], "other") == {}


def test_enrich_caches_and_skips_recompute(tmp_path, monkeypatch):
    reg = Registry(tmp_path / "t.db")
    calls = {"n": 0}

    def fake_get_embeddings(texts, **kw):
        calls["n"] += 1
        return [[float(len(t)), 1.0, 0.0] for t in texts]

    monkeypatch.setattr("acm.pipeline.similarity.get_embeddings", fake_get_embeddings)
    lookup, store = embedding_cache_io(reg, "m1")

    out1 = enrich_records_with_embeddings(
        [_rec("a", "What is X", "def X")], ollama_model="m1",
        cache_lookup=lookup, cache_store=store,
    )
    assert out1[0].features.embedding is not None
    assert calls["n"] == 1

    # Segundo enrich con record fresco pero mismo contenido (== fingerprint) →
    # cache hit, sin nueva llamada a Ollama (E1-3).
    out2 = enrich_records_with_embeddings(
        [_rec("a2", "What is X", "def X")], ollama_model="m1",
        cache_lookup=lookup, cache_store=store,
    )
    assert out2[0].features.embedding == out1[0].features.embedding
    assert calls["n"] == 1


def test_enrich_embeds_front_and_back(monkeypatch):
    captured = {}

    def fake(texts, **kw):
        captured["texts"] = list(texts)
        return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr("acm.pipeline.similarity.get_embeddings", fake)
    enrich_records_with_embeddings([_rec("a", "Question here", "Answer body")], ollama_model="m")
    assert "Question here" in captured["texts"][0]
    assert "Answer body" in captured["texts"][0]  # E1-2: back incluido


def test_embedding_text_combines_fields():
    rec = _rec("a", "Front Q", "Back A")
    assert embedding_text(rec) == "Front Q\nBack A"
    assert embedding_text(_rec("b", "Only front", "")) == "Only front"


def test_enrich_degrades_gracefully_when_ollama_down(monkeypatch):
    monkeypatch.setattr("acm.pipeline.similarity.get_embeddings", lambda texts, **kw: None)
    out = enrich_records_with_embeddings([_rec("a", "Q", "A")], ollama_model="m")
    assert out[0].features.embedding is None  # sin crash, degrada a léxico


# --- E1-1 recuperación kNN ---

def test_knn_clusters_paraphrase_without_lexical_overlap():
    """El caso estrella: dos cards sin NINGÚN solape léxico pero coseno alto."""
    a = _emb(_rec("a", "Alpha bravo charlie delta"), [1.0, 0.0, 0.0])
    b = _emb(_rec("b", "Xray yankee zulu omega"), [0.99, math.sqrt(1 - 0.99 ** 2), 0.0])

    # No comparten block_keys (el léxico nunca los compararía)
    assert not (set(a.features.block_keys) & set(b.features.block_keys))

    clusters, _ = audit_duplicate_records([a, b], cluster_threshold=0.90, similar_threshold=0.75)
    assert len(clusters) == 1
    assert {m.candidate_id for m in clusters[0].members} == {"a", "b"}


def test_without_embeddings_unrelated_text_does_not_cluster():
    a = _rec("a", "Alpha bravo charlie delta")
    b = _rec("b", "Xray yankee zulu omega")
    clusters, _ = audit_duplicate_records([a, b], cluster_threshold=0.90, similar_threshold=0.75)
    assert clusters == []


def test_low_cosine_unrelated_pair_is_not_retrieved():
    """Coseno bajo (no-relacionados) no debe generar candidatos kNN."""
    a = _emb(_rec("a", "Topic one"), [1.0, 0.0])
    b = _emb(_rec("b", "Topic two"), [0.0, 1.0])  # cosine 0.0
    clusters, _ = audit_duplicate_records([a, b], cluster_threshold=0.90, similar_threshold=0.75)
    assert clusters == []
