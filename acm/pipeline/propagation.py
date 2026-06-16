"""E2-3 · Propagación de tags/mazo por kNN.

Una card nueva hereda facetas de sus vecinos más cercanos ya etiquetados,
reusando los embeddings que ya se calcularon para dedup (cero modelo nuevo).
Es el segundo escalón de la escalera de clasificación (RF-C3), antes del LLM
local y de Claude. Solo PROPONE; auto-aplicar queda gateado por alta confianza.
"""

from __future__ import annotations

from collections import defaultdict

from acm.config import Taxonomy
from acm.pipeline.similarity import DuplicateMatch


def propagate_facets_from_neighbors(
    matches: list[DuplicateMatch],
    *,
    taxonomy: Taxonomy,
    min_score: float = 0.70,
    top_k: int = 8,
) -> dict[str, dict]:
    """Vota la faceta más común por categoría entre los vecinos etiquetados.

    Cada vecino vota ponderado por su score de similitud. Solo valores válidos
    en la taxonomía. Devuelve ``{category: {value, confidence, votes}}`` donde
    ``confidence`` = peso del valor ganador / peso total de la categoría.
    """
    valid_categories = set(taxonomy.category_names())
    weighted: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    total_weight: dict[str, float] = defaultdict(float)

    for match in matches[:top_k]:
        if match.score < min_score:
            continue
        for category, value in match.record.scope.summary().items():
            if category not in valid_categories or not taxonomy.is_valid_tag(category, value):
                continue
            weighted[category][value] += match.score
            counts[category][value] += 1
            total_weight[category] += match.score

    result: dict[str, dict] = {}
    for category, value_weights in weighted.items():
        best_value, best_weight = max(value_weights.items(), key=lambda kv: kv[1])
        denom = total_weight[category] or 1.0
        result[category] = {
            "value": best_value,
            "confidence": round(best_weight / denom, 3),
            "votes": counts[category][best_value],
        }
    return result


def apply_high_confidence_propagation(
    classified,
    propagated: dict[str, dict],
    *,
    min_confidence: float = 0.70,
    min_votes: int = 2,
) -> list[str]:
    """Auto-aplica (E2-2) facetas propagadas SOLO para categorías faltantes y con
    alta confianza. Muta `classified.scope`/`tags_resolved`. Devuelve los tags
    nuevos aplicados (para transparencia)."""
    applied: list[str] = []
    for category, info in propagated.items():
        if classified.scope.get(category) is not None:
            continue  # no pisar lo que el determinista ya resolvió
        if info["confidence"] < min_confidence or info["votes"] < min_votes:
            continue
        classified.scope.set(category, info["value"])
        full_tag = f"{category}::{info['value']}"
        if full_tag not in classified.tags_resolved:
            classified.tags_resolved.append(full_tag)
            applied.append(full_tag)
    return applied
