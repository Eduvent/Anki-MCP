"""Validación del motor kNN sobre la colección real (~/.acm/registry.db).

Mide: latencia de la primera corrida (embebe + cachea) vs la segunda (cache hit),
y reporta clusters de duplicados cross-deck encontrados. Valida E1-1/2/3/4/5.

Uso:  PYTHONPATH=. .venv/bin/python scripts/validate_real.py
"""

from __future__ import annotations

import time

from acm.config import load_settings
from acm.pipeline.auditor import _all_indexed_records, _registry_records
from acm.pipeline.similarity import audit_duplicate_records, enrich_records_with_embeddings
from acm.store.registry import Registry, embedding_cache_io


def main() -> None:
    settings = load_settings()
    registry = Registry(settings.db_path_resolved)
    model = settings.acm.ollama_model
    print(f"DB: {settings.db_path_resolved}\nModelo: {model}\n")

    pool = _registry_records(registry) + _all_indexed_records(registry)
    print(f"Pool: {len(pool)} records (registro + índice cross-deck)")
    if not pool:
        print("Sin datos indexados. Corré una auditoría primero.")
        return

    lookup, store = embedding_cache_io(registry, model)

    t0 = time.perf_counter()
    enriched = enrich_records_with_embeddings(
        pool, ollama_url=settings.acm.ollama_url, ollama_model=model,
        cache_lookup=lookup, cache_store=store,
    )
    t_first = time.perf_counter() - t0
    with_emb = sum(1 for r in enriched if r.features.embedding is not None)
    print(f"1ª corrida (embeber+cachear): {t_first:.1f}s · {with_emb}/{len(pool)} con vector")

    t0 = time.perf_counter()
    enrich_records_with_embeddings(
        pool, ollama_url=settings.acm.ollama_url, ollama_model=model,
        cache_lookup=lookup, cache_store=store,
    )
    t_second = time.perf_counter() - t0
    print(f"2ª corrida (cache hit): {t_second:.2f}s  → speedup {t_first / max(t_second, 1e-6):.0f}x")

    t0 = time.perf_counter()
    clusters, metrics = audit_duplicate_records(
        enriched,
        cluster_threshold=settings.acm.cluster_threshold,
        similar_threshold=settings.acm.similar_lookup_threshold,
    )
    t_audit = time.perf_counter() - t0
    print(f"\nClustering kNN: {t_audit:.1f}s")
    print(f"  cards={metrics.cards_scanned} comparaciones={metrics.comparisons_run} "
          f"clusters={metrics.clusters_found}")

    for c in clusters[:8]:
        reasons = ", ".join(c.reason_codes)
        print(f"\n  cluster {c.cluster_id} (score≥{c.score_floor:.2f}, {reasons}):")
        for m in c.members[:4]:
            print(f"    [{m.deck or m.source}] {m.front[:70]!r}")


if __name__ == "__main__":
    main()
