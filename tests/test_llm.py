"""E2-4 · fallback de clasificación con LLM local (Ollama mockeado)."""

from acm.config import Taxonomy
from acm.pipeline import llm as llm_mod
from acm.pipeline.llm import classify_with_llm


def _tax() -> Taxonomy:
    return Taxonomy(vendor=["aws", "azure"], topic=["compute", "storage"])


def test_llm_returns_validated_facets(monkeypatch):
    monkeypatch.setattr(llm_mod, "ollama_generate",
                        lambda *a, **k: 'Sure! {"vendor": "aws", "topic": "compute"}')
    result = classify_with_llm("What is EC2?", "Compute service", _tax(),
                               categories=["vendor", "topic"])
    assert result == {"vendor": "aws", "topic": "compute"}


def test_llm_filters_invalid_taxonomy_values(monkeypatch):
    # 'gcp' no está en la taxonomía → se descarta; 'storage' sí.
    monkeypatch.setattr(llm_mod, "ollama_generate",
                        lambda *a, **k: '{"vendor": "gcp", "topic": "storage"}')
    result = classify_with_llm("Q", "A", _tax(), categories=["vendor", "topic"])
    assert result == {"topic": "storage"}


def test_llm_degrades_when_ollama_down(monkeypatch):
    monkeypatch.setattr(llm_mod, "ollama_generate", lambda *a, **k: None)
    assert classify_with_llm("Q", "A", _tax()) == {}


def test_llm_handles_non_json_response(monkeypatch):
    monkeypatch.setattr(llm_mod, "ollama_generate", lambda *a, **k: "no sé clasificar esto")
    assert classify_with_llm("Q", "A", _tax()) == {}


def test_llm_no_options_returns_empty():
    # Taxonomía vacía → no hay opciones que ofrecer.
    assert classify_with_llm("Q", "A", Taxonomy(), categories=["vendor"]) == {}
