from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Any, Callable, Iterable, Literal

from acm.models import CardScope, ClassifiedCard
from acm.pipeline.embeddings import cosine_similarity, get_embeddings
from acm.pipeline.normalizer import (
    build_semantic_key,
    detect_intent,
    normalize_semantic_text,
    normalize_text,
)


Intent = Literal["definition", "comparison", "unknown"]

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "between",
    "como",
    "con",
    "cual",
    "cuál",
    "dame",
    "de",
    "del",
    "dime",
    "does",
    "el",
    "en",
    "entre",
    "es",
    "explain",
    "for",
    "give",
    "how",
    "is",
    "la",
    "las",
    "lo",
    "los",
    "mean",
    "me",
    "of",
    "por",
    "para",
    "que",
    "qué",
    "s",
    "significa",
    "tell",
    "the",
    "to",
    "un",
    "una",
    "unos",
    "unas",
    "versus",
    "vs",
    "what",
    "y",
}
_BOILERPLATE_TOKENS = {
    "compare",
    "compara",
    "definition",
    "definicion",
    "define",
    "difference",
    "diferencia",
    "explica",
}


@dataclass(frozen=True)
class DuplicateFeatures:
    front_normalized: str
    back_normalized: str
    intent: Intent
    semantic_key: str | None
    content_tokens: tuple[str, ...]
    back_tokens: tuple[str, ...]
    char_trigrams: tuple[str, ...]
    anchor_tokens: tuple[str, ...]
    block_keys: tuple[str, ...]
    trigram_signature: str
    embedding: tuple[float, ...] | None = None


@dataclass(frozen=True)
class DuplicateRecord:
    candidate_id: str
    source: str
    origin_source: str | None
    deck: str | None
    note_type: str
    scope: CardScope
    front: str
    back: str
    fingerprint: str
    created_at: str | None
    anki_note_id: int | None
    features: DuplicateFeatures


@dataclass(frozen=True)
class DuplicateEdge:
    left_id: str
    right_id: str
    score: float
    reason_codes: tuple[str, ...]


@dataclass(frozen=True)
class DuplicateMatch:
    record: DuplicateRecord
    score: float
    reason_codes: tuple[str, ...]


@dataclass(frozen=True)
class DuplicateCluster:
    cluster_id: str
    representative: DuplicateRecord
    members: tuple[DuplicateRecord, ...]
    reason_codes: tuple[str, ...]
    score_floor: float


@dataclass(frozen=True)
class DuplicateMetrics:
    cards_scanned: int
    candidates_generated: int
    comparisons_run: int
    clusters_found: int
    comparison_reduction_pct: float


def make_fingerprint(front_normalized: str, back_normalized: str) -> str:
    raw = f"{front_normalized}||{back_normalized}"
    return hashlib.sha256(raw.encode()).hexdigest()


_SUFFIX_RULES: tuple[tuple[str, int, str | None], ...] = (
    # (suffix, min_stem_length, replacement)
    ("aciones", 3, "ar"),   # configuraciones → configurar
    ("ioning", 3, "ion"),   # provisioning → provision
    ("mente", 3, None),     # completamente → completa
    ("ation", 3, None),     # configuration → configur
    ("ness", 3, None),      # awareness → aware
    ("ment", 3, None),      # management → manage
    ("ting", 3, None),      # networking → network
    ("ning", 3, None),      # running → run  (after -ting so "ting" wins first)
    ("cion", 3, None),      # configuracion → configura
    ("ling", 3, None),      # handling → hand
    ("ices", 5, "ic"),      # services → servic, practices → practic
    ("able", 3, None),      # scalable → scal
    ("ible", 3, None),      # accessible → access
    ("ting", 3, None),      # computing → comput
    ("ing", 3, None),       # running → runn
    ("ion", 3, None),       # encryption → encrypt
    ("ies", 3, "y"),        # policies → policy
    ("ous", 3, None),       # continuous → continu
    ("ado", 3, None),       # configurado → configur
    ("ity", 3, None),       # security → secur
    ("ful", 3, None),       # powerful → power
    ("ly", 3, None),        # directly → direct
    ("ed", 3, None),        # managed → manag
    ("es", 4, None),        # services → servic (min 4 to avoid type→typ)
    ("er", 3, None),        # provider → provid
    ("al", 3, None),        # virtual → virtu
    ("s", 3, None),         # accounts → account
    ("e", 5, None),         # service → servic, configure → configur (min 5 to protect short words)
)


def _stem(token: str) -> str:
    """Suffix-strip ligero para inglés/español técnico."""
    for suffix, min_len, replacement in _SUFFIX_RULES:
        if token.endswith(suffix) and len(token) - len(suffix) >= min_len:
            stem = token[: -len(suffix)]
            if replacement:
                stem += replacement
            return stem
    return token


def _tokenize(text: str) -> list[str]:
    if not text:
        return []
    tokens: list[str] = []
    for token in normalize_semantic_text(text).split():
        if not token:
            continue
        tokens.append(_stem(token))
    return tokens


def _unique_tokens(text: str) -> tuple[str, ...]:
    return tuple(sorted(set(_tokenize(text))))


class TokenStats:
    """Tracks document frequency of tokens across a corpus for IDF-based ranking."""

    def __init__(self) -> None:
        self._doc_freq: dict[str, int] = defaultdict(int)
        self._total_docs: int = 0

    def update(self, tokens: Iterable[str]) -> None:
        self._total_docs += 1
        for token in set(tokens):
            self._doc_freq[token] += 1

    @property
    def total_docs(self) -> int:
        return self._total_docs

    def idf_score(self, token: str) -> float:
        """Higher score = rarer token = more discriminative."""
        df = self._doc_freq.get(token, 0)
        if df == 0 or self._total_docs == 0:
            return 1.0
        return 1.0 / (df / self._total_docs)


# Module-level token stats, built up as records are processed.
_global_token_stats = TokenStats()


def get_token_stats() -> TokenStats:
    return _global_token_stats


def reset_token_stats() -> None:
    global _global_token_stats
    _global_token_stats = TokenStats()


def _anchor_tokens(
    tokens: Iterable[str],
    token_stats: TokenStats | None = None,
) -> tuple[str, ...]:
    informative = {
        token
        for token in tokens
        if len(token) > 1 and token not in _STOPWORDS and token not in _BOILERPLATE_TOKENS
    }
    if token_stats and token_stats.total_docs > 0:
        # Rank by IDF (rarest first), then by length as tiebreaker
        ranked = sorted(
            informative,
            key=lambda t: (-token_stats.idf_score(t), -len(t), t),
        )
    else:
        # Fallback: rank by length (original behavior)
        ranked = sorted(informative, key=lambda token: (-len(token), token))
    return tuple(ranked[:4])


def _char_trigrams(text: str) -> tuple[str, ...]:
    normalized = normalize_semantic_text(text)
    if not normalized:
        return ()
    padded = f"  {normalized}  "
    trigrams = {
        padded[index:index + 3]
        for index in range(max(len(padded) - 2, 1))
        if padded[index:index + 3].strip()
    }
    return tuple(sorted(trigrams))


def _scope_block(scope: CardScope, note_type: str, deck: str | None) -> str | None:
    normalized_note_type = normalize_semantic_text(note_type)
    if scope.facets:
        parts = [
            f"{normalize_semantic_text(category)}={normalize_semantic_text(value)}"
            for category, value in sorted(scope.facets.items())
            if value
        ]
        if parts:
            return f"scope:{'|'.join(parts)}:{normalized_note_type}"
    if deck:
        return f"deck:{normalize_semantic_text(deck)}:{normalized_note_type}"
    return None


def extract_duplicate_features(
    *,
    front: str,
    back: str,
    scope: CardScope,
    note_type: str,
    deck: str | None = None,
    token_stats: TokenStats | None = None,
) -> DuplicateFeatures:
    front_normalized = normalize_text(front)
    back_normalized = normalize_text(back)
    intent = detect_intent(front)
    semantic_key = build_semantic_key(front)
    content_tokens = _unique_tokens(front)
    back_tokens = _unique_tokens(back)
    char_trigrams = _char_trigrams(front)

    # Register tokens in stats for IDF computation
    stats = token_stats or _global_token_stats
    stats.update(content_tokens)

    anchors = _anchor_tokens(content_tokens, stats)

    block_keys: list[str] = []
    scope_key = _scope_block(scope, note_type, deck)
    if scope_key:
        block_keys.append(scope_key)
    if semantic_key:
        block_keys.append(f"semantic:{semantic_key}")
    for token in anchors:
        block_keys.append(f"anchor:{token}")
    for trigram in char_trigrams[:4]:
        block_keys.append(f"tri:{trigram}")

    trigram_signature = "|".join(char_trigrams[:6])

    return DuplicateFeatures(
        front_normalized=front_normalized,
        back_normalized=back_normalized,
        intent=intent,
        semantic_key=semantic_key,
        content_tokens=content_tokens,
        back_tokens=back_tokens,
        char_trigrams=char_trigrams,
        anchor_tokens=anchors,
        block_keys=tuple(sorted(set(block_keys))),
        trigram_signature=trigram_signature,
    )


def build_record_from_classified(
    card: ClassifiedCard,
    *,
    candidate_id: str,
    source: str,
    origin_source: str | None,
    deck: str | None = None,
    created_at: str | None = None,
    anki_note_id: int | None = None,
    token_stats: TokenStats | None = None,
) -> DuplicateRecord:
    features = extract_duplicate_features(
        front=card.front,
        back=card.back,
        scope=card.scope,
        note_type=card.note_type,
        deck=deck,
        token_stats=token_stats,
    )
    return DuplicateRecord(
        candidate_id=candidate_id,
        source=source,
        origin_source=origin_source,
        deck=deck,
        note_type=card.note_type,
        scope=card.scope,
        front=card.front,
        back=card.back,
        fingerprint=card.fingerprint,
        created_at=created_at,
        anki_note_id=anki_note_id,
        features=features,
    )


def build_record_from_fields(
    *,
    candidate_id: str,
    source: str,
    origin_source: str | None,
    front: str,
    back: str,
    scope: CardScope,
    note_type: str,
    deck: str | None = None,
    created_at: str | None = None,
    anki_note_id: int | None = None,
    token_stats: TokenStats | None = None,
) -> DuplicateRecord:
    features = extract_duplicate_features(
        front=front,
        back=back,
        scope=scope,
        note_type=note_type,
        deck=deck,
        token_stats=token_stats,
    )
    return DuplicateRecord(
        candidate_id=candidate_id,
        source=source,
        origin_source=origin_source,
        deck=deck,
        note_type=note_type,
        scope=scope,
        front=front,
        back=back,
        fingerprint=make_fingerprint(
            features.front_normalized,
            features.back_normalized,
        ),
        created_at=created_at,
        anki_note_id=anki_note_id,
        features=features,
    )


def _scope_from_row(row: Any) -> CardScope:
    """Reconstruye un CardScope desde una fila (canónico — usado por CLI y MCP).

    `scope_json` (facets) es la fuente de verdad; las columnas legacy
    scope_vendor/topic/cert se leen solo como fallback si scope_json está vacío.
    """
    raw_scope_json = _optional_row_value(row, "scope_json")

    facets: dict[str, str] = {}
    if raw_scope_json:
        try:
            parsed = json.loads(raw_scope_json)
            if isinstance(parsed, dict):
                facets = {
                    str(key): str(value)
                    for key, value in parsed.items()
                    if value is not None and str(value).strip()
                }
        except json.JSONDecodeError:
            facets = {}

    return CardScope(
        vendor=_optional_row_value(row, "scope_vendor"),
        topic=_optional_row_value(row, "scope_topic"),
        cert=_optional_row_value(row, "scope_cert"),
        facets=facets,
    )


def build_record_from_registry_row(
    row: Any,
    token_stats: TokenStats | None = None,
) -> DuplicateRecord:
    scope = _scope_from_row(row)
    note_type = row["note_type"] or "Basic"
    features = extract_duplicate_features(
        front=row["front_original"],
        back=row["back_original"],
        scope=scope,
        note_type=note_type,
        deck=None,
        token_stats=token_stats,
    )
    return DuplicateRecord(
        candidate_id=row["id"],
        source="registry",
        origin_source=row["source"],
        deck=None,
        note_type=note_type,
        scope=scope,
        front=row["front_original"],
        back=row["back_original"],
        fingerprint=row["fingerprint"],
        created_at=row["created_at"],
        anki_note_id=row["anki_note_id"],
        features=features,
    )


def _has_cached_features(row: Any) -> bool:
    """Check if the index row has full cached features (from the new schema)."""
    try:
        return bool(row["content_tokens"] or row["block_keys"])
    except (IndexError, KeyError):
        return False


def _optional_row_value(row: Any, key: str, default: Any = None) -> Any:
    """Read an optional field from dict-like rows and sqlite3.Row objects."""
    if isinstance(row, dict):
        return row.get(key, default)
    try:
        return row[key]
    except (IndexError, KeyError, TypeError):
        return default


def _features_from_index_cache(row: Any, scope: CardScope, deck: str) -> DuplicateFeatures:
    """Reconstruct DuplicateFeatures from cached columns without recomputing."""
    content_tokens_raw = _optional_row_value(row, "content_tokens", "")
    back_tokens_raw = _optional_row_value(row, "back_tokens", "")
    char_trigrams_raw = _optional_row_value(row, "char_trigrams", "")
    anchor_tokens_raw = _optional_row_value(row, "anchor_tokens", "")
    block_keys_raw = _optional_row_value(row, "block_keys", "")

    content_tokens = tuple(content_tokens_raw.split()) if content_tokens_raw else ()
    back_tokens = tuple(back_tokens_raw.split()) if back_tokens_raw else ()
    char_trigrams = tuple(char_trigrams_raw.split("|")) if char_trigrams_raw else ()
    anchor_tokens = tuple(anchor_tokens_raw.split()) if anchor_tokens_raw else ()
    block_keys = tuple(block_keys_raw.split("|")) if block_keys_raw else ()

    # E1-3/E1-4: los embeddings ya NO se leen de la columna por-nota (que podía
    # quedar con vectores de un modelo viejo y otra dimensión). La única fuente
    # es el embedding_cache model-aware, que enrich_records_with_embeddings
    # consulta por fingerprint. Acá el record nace sin vector y se enriquece.
    embedding: tuple[float, ...] | None = None

    return DuplicateFeatures(
        front_normalized=row["front_normalized"],
        back_normalized=row["back_normalized"],
        intent=_optional_row_value(row, "intent", "unknown") or "unknown",
        semantic_key=_optional_row_value(row, "semantic_key"),
        content_tokens=content_tokens,
        back_tokens=back_tokens,
        char_trigrams=char_trigrams,
        anchor_tokens=anchor_tokens,
        block_keys=block_keys,
        trigram_signature=_optional_row_value(row, "trigram_signature", ""),
        embedding=embedding,
    )


def build_record_from_index_row(row: Any) -> DuplicateRecord:
    scope = _scope_from_row(row)
    note_type = _optional_row_value(row, "note_type", "Basic") or "Basic"
    if _has_cached_features(row):
        features = _features_from_index_cache(row, scope, row["deck_name"])
    else:
        # Fallback: recompute for rows from old schema without cached features
        features = extract_duplicate_features(
            front=row["front_original"],
            back=row["back_original"],
            scope=scope,
            note_type=note_type,
            deck=row["deck_name"],
        )
    return DuplicateRecord(
        candidate_id=f"anki:{row['note_id']}",
        source="anki",
        origin_source="anki",
        deck=row["deck_name"],
        note_type=note_type,
        scope=scope,
        front=row["front_original"],
        back=row["back_original"],
        fingerprint=row["fingerprint"],
        created_at=_optional_row_value(row, "indexed_at"),
        anki_note_id=row["note_id"],
        features=features,
    )


def record_to_index_entry(record: DuplicateRecord) -> dict[str, Any]:
    return {
        "note_id": record.anki_note_id,
        "deck_name": record.deck,
        "note_type": record.note_type,
        "front_original": record.front,
        "back_original": record.back,
        "front_normalized": record.features.front_normalized,
        "back_normalized": record.features.back_normalized,
        "fingerprint": record.fingerprint,
        "scope_json": json.dumps(record.scope.summary(), sort_keys=True),
        "intent": record.features.intent,
        "semantic_key": record.features.semantic_key,
        "anchor_tokens": " ".join(record.features.anchor_tokens),
        "trigram_signature": record.features.trigram_signature,
        "content_tokens": " ".join(record.features.content_tokens),
        "back_tokens": " ".join(record.features.back_tokens),
        "char_trigrams": "|".join(record.features.char_trigrams),
        "block_keys": "|".join(record.features.block_keys),
        "embedding_vector": json.dumps(list(record.features.embedding)) if record.features.embedding else "",
        "indexed_at": record.created_at or "",
    }


def embedding_text(record: DuplicateRecord) -> str:
    """Texto a embeber para dedup semántica: front + back (E1-2).

    Antes solo se embebía el ``front`` y se perdía la similitud de las
    respuestas (misma pregunta con respuesta reformulada). Combinar ambos
    campos captura las dos señales en un único vector.
    """
    front = (record.front or "").strip()
    back = (record.back or "").strip()
    return f"{front}\n{back}".strip() if back else front


def _record_with_embedding(record: DuplicateRecord, vector: tuple[float, ...]) -> DuplicateRecord:
    return replace(record, features=replace(record.features, embedding=vector))


def enrich_records_with_embeddings(
    records: list[DuplicateRecord],
    *,
    ollama_url: str = "http://localhost:11434",
    ollama_model: str = "qwen3-embedding:8b",
    batch_size: int = 64,
    cache_lookup: Callable[[list[str]], dict[str, list[float]]] | None = None,
    cache_store: Callable[[dict[str, list[float]]], None] | None = None,
) -> list[DuplicateRecord]:
    """Asegura que cada record tenga embedding (front+back), reusando cache.

    Pasos por ``fingerprint``: (1) ``cache_lookup``; (2) Ollama para los que
    falten; (3) ``cache_store`` de los nuevos. Si Ollama no responde, devuelve
    lo que tenga (incluidos cache hits) sin fallar — la dedup degrada a léxico
    de forma transparente (E0-3 lo reporta). Cachear por fingerprint evita
    re-embeber todo el registro en cada auditoría (E1-3).
    """
    pending = [i for i, rec in enumerate(records) if rec.features.embedding is None]
    if not pending:
        return records

    enriched = list(records)

    # 1. Cache por fingerprint (reuso entre registro, índice y batch).
    cached: dict[str, list[float]] = {}
    if cache_lookup is not None:
        cached = cache_lookup([records[i].fingerprint for i in pending])

    still_pending: list[int] = []
    for i in pending:
        vec = cached.get(records[i].fingerprint)
        if vec:
            enriched[i] = _record_with_embedding(records[i], tuple(vec))
        else:
            still_pending.append(i)

    if not still_pending:
        return enriched

    # 2. Calcular los faltantes vía Ollama, en batches.
    texts = [embedding_text(records[i]) for i in still_pending]
    all_embeddings: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        result = get_embeddings(
            batch, ollama_url=ollama_url, model=ollama_model, timeout=120.0,
        )
        if result is None:
            return enriched  # Ollama caído → parcial; degrada a léxico
        all_embeddings.extend(result)

    if len(all_embeddings) != len(texts):
        return enriched

    # 3. Aplicar y cachear los nuevos por fingerprint.
    newly_computed: dict[str, list[float]] = {}
    for offset, i in enumerate(still_pending):
        vec = all_embeddings[offset]
        enriched[i] = _record_with_embedding(records[i], tuple(vec))
        newly_computed.setdefault(records[i].fingerprint, vec)

    if cache_store is not None and newly_computed:
        cache_store(newly_computed)

    return enriched


def _jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set or not right_set:
        return 0.0
    intersection = len(left_set & right_set)
    union = len(left_set | right_set)
    return intersection / union if union else 0.0


# E1-6: calibración coseno → escala léxica [0,1]. Anclas del spike E1-0 con
# qwen3-embedding:0.6b (ver SPIKE_EMBEDDINGS.md): paráfrasis ES/EN ~0.55-0.64,
# no-relacionados ~0.07-0.20. El coseno crudo NO está en la misma escala que las
# señales léxicas (0.80-1.0); sin calibrar, cluster_threshold=0.90 jamás
# agruparía una paráfrasis a coseno 0.60. Monótona; interpola lineal entre anclas.
# Si cambiás de modelo de embeddings, recalibrá con `scripts/spike_knn.py`.
_COSINE_CALIBRATION: tuple[tuple[float, float], ...] = (
    (0.20, 0.00),   # techo de no-relacionados → 0
    (0.40, 0.75),   # "similar"  (similar_lookup_threshold)
    (0.55, 0.90),   # "cluster"  (piso de paráfrasis → duplicado fuerte)
    (0.68, 0.97),
    (1.00, 1.00),
)
_EMBEDDING_REASON_FLOOR = 0.45  # coseno crudo desde el que se reporta embedding_match


def _interpolate(table: tuple[tuple[float, float], ...], x: float, *, invert: bool) -> float:
    lo_i, hi_i = (1, 0) if invert else (0, 1)
    if x <= table[0][lo_i]:
        return table[0][hi_i]
    if x >= table[-1][lo_i]:
        return table[-1][hi_i]
    for left_pt, right_pt in zip(table, table[1:]):
        if left_pt[lo_i] <= x <= right_pt[lo_i]:
            span = right_pt[lo_i] - left_pt[lo_i]
            t = (x - left_pt[lo_i]) / span if span else 0.0
            return left_pt[hi_i] + t * (right_pt[hi_i] - left_pt[hi_i])
    return table[-1][hi_i]


def calibrate_cosine(cosine: float) -> float:
    """Mapea coseno crudo a la escala léxica [0,1] (E1-6)."""
    return _interpolate(_COSINE_CALIBRATION, cosine, invert=False)


def cosine_floor_for(calibrated_target: float) -> float:
    """Inverso de calibrate_cosine: menor coseno crudo cuyo score ≥ target.

    Sirve de piso de recuperación kNN en espacio de coseno crudo, para no perder
    paráfrasis que sí superan el umbral una vez calibradas.
    """
    return _interpolate(_COSINE_CALIBRATION, calibrated_target, invert=True)


def compare_records(
    left: DuplicateRecord,
    right: DuplicateRecord,
    *,
    similar_threshold: float,
) -> DuplicateEdge | None:
    """Compara dos records usando señales discretas.

    Jerarquía de señales (de más fuerte a más débil):
      1.00  exact_fingerprint  — SHA256 idéntico (front+back)
      0.98  exact_front        — front normalizado idéntico
      0.95  semantic_key_match — misma clave semántica (definition::X, comparison::X::Y)
      0.90  strong_lexical     — front tokens Jaccard ≥ 0.70
      0.80  moderate_lexical   — front tokens Jaccard ≥ 0.50 AND trigram Jaccard ≥ 0.40
      raw   fallback           — Jaccard ponderado para scores bajos

    La señal más fuerte gana. Se reportan todas las señales que apliquen.
    """
    reason_codes: list[str] = []

    # --- Señales exactas ---
    is_exact_fingerprint = left.fingerprint == right.fingerprint
    is_exact_front = (
        left.features.front_normalized
        and left.features.front_normalized == right.features.front_normalized
    )
    is_semantic_match = (
        left.features.semantic_key
        and left.features.semantic_key == right.features.semantic_key
    )

    # --- Métricas léxicas ---
    front_token_jaccard = _jaccard(left.features.content_tokens, right.features.content_tokens)
    trigram_jaccard = _jaccard(left.features.char_trigrams, right.features.char_trigrams)
    back_token_jaccard = _jaccard(left.features.back_tokens, right.features.back_tokens)

    is_strong_lexical = front_token_jaccard >= 0.70
    is_moderate_lexical = front_token_jaccard >= 0.50 and trigram_jaccard >= 0.40

    # --- Embedding cosine similarity ---
    has_embeddings = (
        left.features.embedding is not None and right.features.embedding is not None
    )
    embedding_cosine = 0.0
    if has_embeddings:
        embedding_cosine = cosine_similarity(left.features.embedding, right.features.embedding)

    # --- Score léxico por señal más fuerte (independiente de embeddings) ---
    if is_exact_fingerprint:
        lexical_score = 1.0
    elif is_exact_front:
        lexical_score = 0.98
    elif is_semantic_match:
        lexical_score = 0.95
    elif is_strong_lexical:
        lexical_score = 0.90
    elif is_moderate_lexical:
        lexical_score = 0.80
    else:
        # Fallback: Jaccard ponderado para scores bajos
        lexical_score = (
            0.65 * front_token_jaccard
            + 0.25 * trigram_jaccard
            + 0.10 * back_token_jaccard
        )

    # --- Combinar señales (E0-2 + E1-6) ---
    # El coseno se CALIBRA a la escala léxica antes de combinar (E1-6), y los
    # embeddings SUMAN recall (paráfrasis sin solape léxico) pero NUNCA arrastran
    # una señal léxica fuerte por debajo de sí misma (E0-2: max(), no reemplazo).
    calibrated_embedding = calibrate_cosine(embedding_cosine) if has_embeddings else 0.0
    score = max(lexical_score, calibrated_embedding)

    # --- Reportar todas las señales que apliquen ---
    if is_exact_fingerprint:
        reason_codes.append("exact_fingerprint")
    if is_exact_front:
        reason_codes.append("exact_front")
    if is_semantic_match:
        reason_codes.append("semantic_key_match")
    if has_embeddings and embedding_cosine >= _EMBEDDING_REASON_FLOOR:
        reason_codes.append("embedding_match")
    if front_token_jaccard >= 0.50:
        reason_codes.append("token_overlap")
    if trigram_jaccard >= 0.40:
        reason_codes.append("trigram_overlap")

    score = min(score, 1.0)
    if score < similar_threshold:
        return None

    if not reason_codes:
        reason_codes.append("low_similarity")

    return DuplicateEdge(
        left_id=left.candidate_id,
        right_id=right.candidate_id,
        score=round(score, 4),
        reason_codes=tuple(sorted(set(reason_codes))),
    )


def _build_candidate_pairs(records: list[DuplicateRecord]) -> tuple[set[tuple[int, int]], int]:
    block_index: dict[str, set[int]] = defaultdict(set)
    pairs: set[tuple[int, int]] = set()
    raw_candidates = 0

    for index, record in enumerate(records):
        for key in record.features.block_keys:
            others = block_index[key]
            raw_candidates += len(others)
            for other_index in others:
                pairs.add((other_index, index))
            others.add(index)

    return pairs, raw_candidates


def _exact_candidate_pairs(records: list[DuplicateRecord]) -> set[tuple[int, int]]:
    """Atajo exacto (E1-1, Opción A): pares con mismo fingerprint o front idéntico.

    Independiente del léxico pesado para que sobreviva a E1-7 (retiro del léxico).
    """
    by_fingerprint: dict[str, list[int]] = defaultdict(list)
    by_front: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_fingerprint[record.fingerprint].append(index)
        if record.features.front_normalized:
            by_front[record.features.front_normalized].append(index)

    pairs: set[tuple[int, int]] = set()
    for group in (*by_fingerprint.values(), *by_front.values()):
        for a in range(len(group)):
            for b in range(a + 1, len(group)):
                pairs.add((min(group[a], group[b]), max(group[a], group[b])))
    return pairs


def _embedding_candidate_pairs(
    records: list[DuplicateRecord], *, floor: float
) -> set[tuple[int, int]]:
    """Recuperación kNN por coseno (E1-1): pares con coseno CRUDO ≥ floor.

    Es el mecanismo de candidatos de la Opción A: captura paráfrasis sin solape
    léxico (el caso estrella de los embeddings), que el blocking léxico nunca
    llegaba a comparar. Pre-normaliza a vectores unitarios → coseno = producto
    punto (evita recomputar normas por par). O(M²·dim) sobre los M records con
    vector; suficiente para colecciones de cientos/miles (optimizable con un
    índice ANN si crece mucho).
    """
    unit: list[tuple[int, list[float]]] = []
    for index, record in enumerate(records):
        vector = record.features.embedding
        if vector is None:
            continue
        magnitude = math.sqrt(sum(component * component for component in vector))
        if magnitude == 0.0:
            continue
        unit.append((index, [component / magnitude for component in vector]))

    pairs: set[tuple[int, int]] = set()
    for a in range(len(unit)):
        index_a, vector_a = unit[a]
        for b in range(a + 1, len(unit)):
            index_b, vector_b = unit[b]
            dot = sum(x * y for x, y in zip(vector_a, vector_b))
            if dot >= floor:
                pairs.add((min(index_a, index_b), max(index_a, index_b)))
    return pairs


def _record_sort_key(record: DuplicateRecord) -> tuple[int, int | str, str]:
    if record.source == "anki":
        return (0, record.anki_note_id or 0, record.candidate_id)
    return (1, record.created_at or "", record.candidate_id)


def _make_cluster_id(records: Iterable[DuplicateRecord]) -> str:
    raw = "||".join(sorted(record.candidate_id for record in records))
    return hashlib.sha1(raw.encode()).hexdigest()[:12]


def audit_duplicate_records(
    records: list[DuplicateRecord],
    *,
    cluster_threshold: float,
    similar_threshold: float,
) -> tuple[list[DuplicateCluster], DuplicateMetrics]:
    # Note: records already have features with anchor tokens computed.
    # IDF stats are accumulated in _global_token_stats as records are built.
    # E1-1 (Opción A): kNN por coseno es el recuperador principal cuando hay
    # embeddings; el blocking léxico queda como fallback y se suma el atajo
    # exacto (fingerprint/front). Unión = recall estrictamente mayor.
    lexical_pairs, raw_candidates = _build_candidate_pairs(records)
    pairs = (
        lexical_pairs
        | _embedding_candidate_pairs(records, floor=cosine_floor_for(similar_threshold))
        | _exact_candidate_pairs(records)
    )
    strong_edges: list[DuplicateEdge] = []

    for left_index, right_index in sorted(pairs):
        edge = compare_records(
            records[left_index],
            records[right_index],
            similar_threshold=similar_threshold,
        )
        if edge and edge.score >= cluster_threshold:
            strong_edges.append(edge)

    adjacency: dict[str, set[str]] = defaultdict(set)
    for edge in strong_edges:
        adjacency[edge.left_id].add(edge.right_id)
        adjacency[edge.right_id].add(edge.left_id)

    by_id = {record.candidate_id: record for record in records}
    visited: set[str] = set()
    strong_edge_map = {
        tuple(sorted((edge.left_id, edge.right_id))): edge
        for edge in strong_edges
    }
    clusters: list[DuplicateCluster] = []

    for record in sorted(records, key=_record_sort_key):
        if record.candidate_id in visited or record.candidate_id not in adjacency:
            continue

        stack = [record.candidate_id]
        component_ids: list[str] = []
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            component_ids.append(current)
            stack.extend(sorted(adjacency[current] - visited))

        if len(component_ids) < 2:
            continue

        members = sorted((by_id[item_id] for item_id in component_ids), key=_record_sort_key)
        component_edges = [
            edge
            for pair, edge in strong_edge_map.items()
            if pair[0] in component_ids and pair[1] in component_ids
        ]
        reason_codes = sorted({
            reason_code
            for edge in component_edges
            for reason_code in edge.reason_codes
        })
        score_floor = min(edge.score for edge in component_edges)
        clusters.append(
            DuplicateCluster(
                cluster_id=_make_cluster_id(members),
                representative=members[0],
                members=tuple(members),
                reason_codes=tuple(reason_codes),
                score_floor=round(score_floor, 4),
            )
        )

    total_pairs = len(records) * (len(records) - 1) // 2
    reduction = 0.0
    if total_pairs:
        reduction = round((1 - (len(pairs) / total_pairs)) * 100, 2)

    metrics = DuplicateMetrics(
        cards_scanned=len(records),
        candidates_generated=raw_candidates,
        comparisons_run=len(pairs),
        clusters_found=len(clusters),
        comparison_reduction_pct=reduction,
    )
    return clusters, metrics


def find_similar_records(
    query: DuplicateRecord,
    records: list[DuplicateRecord],
    *,
    similar_threshold: float,
    limit: int = 20,
) -> list[DuplicateMatch]:
    candidate_indexes: set[int] = set()

    # 1. Recuperación kNN por coseno (E1-1, mecanismo principal con embeddings).
    #    El piso está en espacio de coseno crudo (E1-6), no en la escala léxica.
    if query.features.embedding is not None:
        raw_floor = cosine_floor_for(similar_threshold)
        for index, record in enumerate(records):
            if record.features.embedding is None:
                continue
            if cosine_similarity(query.features.embedding, record.features.embedding) >= raw_floor:
                candidate_indexes.add(index)

    # 2. Atajo exacto (fingerprint / front idéntico) — siempre.
    for index, record in enumerate(records):
        if record.fingerprint == query.fingerprint or (
            query.features.front_normalized
            and record.features.front_normalized == query.features.front_normalized
        ):
            candidate_indexes.add(index)

    # 3. Fallback léxico por block_keys (Opción B; recall si no hay embeddings).
    block_index: dict[str, set[int]] = defaultdict(set)
    for index, record in enumerate(records):
        for key in record.features.block_keys:
            block_index[key].add(index)
    for key in query.features.block_keys:
        candidate_indexes.update(block_index.get(key, set()))

    matches: list[DuplicateMatch] = []
    for index in sorted(candidate_indexes):
        record = records[index]
        edge = compare_records(query, record, similar_threshold=similar_threshold)
        if not edge:
            continue
        matches.append(
            DuplicateMatch(
                record=record,
                score=edge.score,
                reason_codes=edge.reason_codes,
            )
        )

    matches.sort(key=lambda match: (-match.score, _record_sort_key(match.record)))
    return matches[:limit]


def serialize_record(record: DuplicateRecord) -> dict[str, Any]:
    return {
        "id": record.candidate_id,
        "source": record.source,
        "origin_source": record.origin_source,
        "deck": record.deck,
        "note_type": record.note_type,
        "front": record.front,
        "back_excerpt": record.back[:160],
        "scope": record.scope.summary(),
        "intent": record.features.intent,
        "semantic_key": record.features.semantic_key,
    }


def serialize_cluster(cluster: DuplicateCluster) -> dict[str, Any]:
    return {
        "cluster_id": cluster.cluster_id,
        "representative": serialize_record(cluster.representative),
        "members": [serialize_record(member) for member in cluster.members],
        "reason_codes": list(cluster.reason_codes),
        "score_floor": cluster.score_floor,
    }


def serialize_match(match: DuplicateMatch) -> dict[str, Any]:
    return {
        **serialize_record(match.record),
        "score": match.score,
        "reason_codes": list(match.reason_codes),
    }


def serialize_metrics(metrics: DuplicateMetrics) -> dict[str, Any]:
    return {
        "cards_scanned": metrics.cards_scanned,
        "candidates_generated": metrics.candidates_generated,
        "comparisons_run": metrics.comparisons_run,
        "clusters_found": metrics.clusters_found,
        "comparison_reduction_pct": metrics.comparison_reduction_pct,
    }
