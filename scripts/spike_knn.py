"""E1-0 · Spike: validar embeddings-como-recuperador (kNN) antes de retirar el léxico.

Mide sobre pares reales de flashcards (paráfrasis ES/EN sin solape léxico, y
pares no relacionados):
  1. ¿El coseno separa paráfrasis de no-relacionados? (recall semántico)
  2. ¿El blocking léxico habría generado esos pares como candidatos? (lo que se pierde hoy)
  3. Latencia de embedding (batch) y por-card.

Uso:  .venv/bin/python scripts/spike_knn.py
"""

from __future__ import annotations

import time

from acm.config import load_settings
from acm.models import CardScope
from acm.pipeline.embeddings import cosine_similarity, get_embeddings
from acm.pipeline.similarity import build_record_from_fields, embedding_text

# Paráfrasis: mismo significado, palabras distintas, BAJO solape léxico.
PARAPHRASES = [
    (("What is a CDN?", "A content delivery network caches content near users."),
     ("Explain how edge caching distributes assets close to clients.", "It serves copies from points of presence to cut latency.")),
    (("¿Qué hace un balanceador de carga?", "Reparte el tráfico entre varios servidores."),
     ("How does traffic get spread across multiple backends?", "A device distributes requests so no single host is overwhelmed.")),
    (("Define idempotencia en APIs REST", "La misma petición repetida no cambia el resultado."),
     ("What does it mean that PUT can be repeated safely?", "Sending it many times has the same effect as once.")),
    (("What is horizontal scaling?", "Adding more machines to handle load."),
     ("Cómo se escala agregando nodos en lugar de agrandar uno", "Se suman instancias para repartir la carga.")),
]

# No relacionados: temas distintos.
UNRELATED = [
    (("What is a CDN?", "A content delivery network caches content near users."),
     ("How do you make sourdough bread?", "Mix flour and water, ferment, then bake.")),
    (("Define idempotencia en APIs REST", "La misma petición repetida no cambia el resultado."),
     ("¿Cuál es la capital de Francia?", "París.")),
    (("What is horizontal scaling?", "Adding more machines to handle load."),
     ("What year did the Roman Empire fall?", "476 AD in the West.")),
]


def _rec(front: str, back: str):
    return build_record_from_fields(
        candidate_id=front[:12], source="spike", origin_source="spike",
        front=front, back=back, scope=CardScope(), note_type="Basic",
    )


def _shares_block_key(a_front, a_back, b_front, b_back) -> bool:
    a = _rec(a_front, a_back)
    b = _rec(b_front, b_back)
    return bool(set(a.features.block_keys) & set(b.features.block_keys))


def _evaluate(model: str, url: str, *, query_prefix: str = "", doc_prefix: str = "") -> None:
    """Embebe los pares con `model` (con prefijos opcionales) y reporta separación."""
    texts: list[str] = []
    for (af, ab), (bf, bb) in PARAPHRASES + UNRELATED:
        texts.append(doc_prefix + embedding_text(_rec(af, ab)))
        texts.append(doc_prefix + embedding_text(_rec(bf, bb)))

    # calentar (cargar modelo) y luego medir
    get_embeddings(texts[:1], ollama_url=url, model=model, timeout=180.0)
    t0 = time.perf_counter()
    vectors = get_embeddings(texts, ollama_url=url, model=model, timeout=180.0)
    elapsed = time.perf_counter() - t0
    label = f"{model}{' +prefix' if doc_prefix else ''}"
    if vectors is None:
        print(f"[{label}] ERROR: sin respuesta (¿modelo pulled?)\n")
        return

    para_cos = [cosine_similarity(vectors[2 * i], vectors[2 * i + 1]) for i in range(len(PARAPHRASES))]
    base = len(PARAPHRASES)
    unrel_cos = [cosine_similarity(vectors[2 * (base + j)], vectors[2 * (base + j) + 1]) for j in range(len(UNRELATED))]
    para_min, unrel_max = min(para_cos), max(unrel_cos)
    gap = para_min - unrel_max
    print(f"[{label}]  dim={len(vectors[0])}  {elapsed / len(texts) * 1000:.0f} ms/texto (warm)")
    print(f"    paráfrasis  min={para_min:.3f} avg={sum(para_cos)/len(para_cos):.3f}  {[round(c,2) for c in para_cos]}")
    print(f"    no-relac.   max={unrel_max:.3f} avg={sum(unrel_cos)/len(unrel_cos):.3f}  {[round(c,2) for c in unrel_cos]}")
    print(f"    separación={gap:+.3f}  umbral_sugerido~{(para_min + unrel_max) / 2:.2f}  "
          f"{'VIABLE' if gap > 0.05 else 'DÉBIL'}\n")


def main() -> None:
    settings = load_settings()
    url = settings.acm.ollama_url
    lexical_missed = sum(
        1 for (af, ab), (bf, bb) in PARAPHRASES if not _shares_block_key(af, ab, bf, bb)
    )
    print(f"Pares: {len(PARAPHRASES)} paráfrasis (ES/EN) + {len(UNRELATED)} no-relac.")
    print(f"Paráfrasis que el blocking léxico NO compararía: {lexical_missed}/{len(PARAPHRASES)}\n")

    # nomic requiere prefijos de tarea (search_document:) para similitud óptima.
    _evaluate("nomic-embed-text", url)
    _evaluate("nomic-embed-text", url, doc_prefix="search_document: ")
    _evaluate("qwen3-embedding:0.6b", url)
    _evaluate("qwen3-embedding:8b", url)


if __name__ == "__main__":
    main()
