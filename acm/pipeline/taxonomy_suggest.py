"""E2-6 · Taxonomía auto-sugerida.

Cuando emergen clusters de cards SIN tag que comparten tema, el sistema PROPONE
un tag/valor nuevo (no lo crea solo — RF-C5 / decisión #3). Reusa el motor de
clustering por embeddings; la etiqueta candidata sale de los tokens comunes.
"""

from __future__ import annotations

from collections import Counter

from acm.config import ProfileConfig, Taxonomy
from acm.pipeline.normalizer import normalize_semantic_text
from acm.pipeline.similarity import _STOPWORDS, DuplicateRecord, audit_duplicate_records


def _is_untagged(record: DuplicateRecord, required: list[str]) -> bool:
    """Una card está 'sin clasificar' si le falta alguna categoría requerida."""
    facets = record.scope.summary()
    return any(category not in facets for category in required)


def _candidate_label(records: tuple[DuplicateRecord, ...]) -> str | None:
    """Token significativo más frecuente entre los fronts del cluster."""
    counter: Counter[str] = Counter()
    for record in records:
        seen = set()
        for token in normalize_semantic_text(record.front).split():
            if len(token) <= 3 or token in _STOPWORDS or token in seen:
                continue
            seen.add(token)
            counter[token] += 1
    if not counter:
        return None
    label, count = counter.most_common(1)[0]
    # Solo proponer si el token aparece en al menos la mitad del cluster.
    return label if count >= max(2, len(records) // 2) else None


def suggest_taxonomy_proposals(
    pool: list[DuplicateRecord],
    *,
    taxonomy: Taxonomy,
    profile: ProfileConfig,
    cluster_threshold: float,
    similar_threshold: float,
    suggest_category: str = "topic",
    min_cluster_size: int = 3,
) -> list[dict]:
    """Propone tags nuevos a partir de clusters de cards sin clasificar.

    Devuelve una lista de propuestas {suggested_category, suggested_value,
    is_new, cluster_size, sample_fronts}. NO modifica la taxonomía.
    """
    required = profile.required_categories or ["topic"]
    untagged = [record for record in pool if _is_untagged(record, required)]
    if len(untagged) < min_cluster_size:
        return []

    clusters, _ = audit_duplicate_records(
        untagged,
        cluster_threshold=similar_threshold,  # más laxo: agrupa por TEMA, no dup
        similar_threshold=similar_threshold,
    )

    existing_values = set(taxonomy.values_for(suggest_category))
    proposals: list[dict] = []
    for cluster in clusters:
        if len(cluster.members) < min_cluster_size:
            continue
        label = _candidate_label(cluster.members)
        if not label:
            continue
        proposals.append({
            "suggested_category": suggest_category,
            "suggested_value": label,
            "is_new": label not in existing_values,
            "cluster_size": len(cluster.members),
            "sample_fronts": [member.front[:80] for member in cluster.members[:5]],
        })

    proposals.sort(key=lambda proposal: -proposal["cluster_size"])
    return proposals
