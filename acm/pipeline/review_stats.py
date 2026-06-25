from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace
from statistics import mean
from typing import Any

from acm.anki.client import AnkiConnectClient
from acm.config import Settings, Taxonomy, ProfileConfig
from acm.models import CardScope
from acm.pipeline.classifier import classify_fields
from acm.pipeline.normalizer import normalize_text, strip_html_for_display
from acm.pipeline.similarity import (
    DuplicateCluster,
    DuplicateRecord,
    audit_duplicate_records,
    build_record_from_fields,
    enrich_records_with_embeddings,
    make_fingerprint,
)
from acm.store.registry import Registry, embedding_cache_io


def _field_value(fields: dict, name: str) -> str:
    value = fields.get(name, "") if isinstance(fields, dict) else ""
    if isinstance(value, dict):
        return str(value.get("value", ""))
    return str(value or "")


def _first_field_value(fields: dict, preferred_names: tuple[str, ...], *, fallback_index: int) -> str:
    """Extrae campos de Anki tolerando modelos localizados/personalizados."""
    if not isinstance(fields, dict) or not fields:
        return ""

    lower_to_name = {str(name).strip().lower(): name for name in fields}
    for preferred in preferred_names:
        real_name = lower_to_name.get(preferred.lower())
        if real_name:
            return _field_value(fields, real_name)

    values = list(fields.values())
    if 0 <= fallback_index < len(values):
        raw = values[fallback_index]
        if isinstance(raw, dict):
            return str(raw.get("value", ""))
        return str(raw or "")
    return ""


def _excerpt(text: str, chars: int = 160) -> str:
    clean = strip_html_for_display(text)
    return clean[:chars]


def _is_leech(tags: list[str], lapses: int, min_lapses: int) -> bool:
    normalized_tags = {str(tag).strip().lower() for tag in tags}
    return "leech" in normalized_tags or lapses >= min_lapses


def _review_rows(raw: Any, card_id: int) -> list[list[Any]]:
    if isinstance(raw, dict):
        rows = raw.get(str(card_id), raw.get(card_id, []))
    else:
        rows = raw or []
    return rows if isinstance(rows, list) else []


def _review_ease(row: Any) -> int | None:
    if isinstance(row, dict):
        value = row.get("ease")
    elif isinstance(row, (list, tuple)) and len(row) > 3:
        value = row[3]
    else:
        value = None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _review_time_ms(row: Any) -> int | None:
    if isinstance(row, dict):
        value = row.get("time") or row.get("reviewTime")
    elif isinstance(row, (list, tuple)) and len(row) > 7:
        value = row[7]
    else:
        value = None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _unique_preserve(ids: list[int]) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for value in ids:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def build_card_query(deck: str | None = None, tag: str | None = None, query: str | None = None) -> str:
    parts: list[str] = []
    if query:
        parts.append(query.strip())
    if deck:
        parts.append(f'"deck:{deck}"')
    if tag:
        parts.append(f'tag:{tag}')
    return " ".join(parts) if parts else "*"


def review_stats(
    client: AnkiConnectClient,
    *,
    deck: str | None = None,
    tag: str | None = None,
    query: str | None = None,
    min_lapses: int = 8,
    limit: int = 50,
    include_cards: bool = True,
) -> dict:
    """Extrae desempeño real de repaso desde AnkiConnect, sin LLM."""
    search = build_card_query(deck=deck, tag=tag, query=query)
    card_ids = _unique_preserve([int(cid) for cid in client.find_cards(search)])

    if not card_ids:
        return {
            "query": search,
            "cards_scanned": 0,
            "notes_scanned": 0,
            "summary": {"leeches": 0, "suspended": 0, "again_count": 0, "avg_review_time_ms": None},
            "cards": [],
        }

    infos = client.cards_info(card_ids)
    info_by_card = {int(info.get("cardId", info.get("card", 0))): info for info in infos}
    note_ids = _unique_preserve([
        int(info.get("note", info.get("noteId"))) for info in infos if info.get("note", info.get("noteId")) is not None
    ])
    notes = client.get_notes_info(note_ids) if note_ids else []
    note_by_id = {int(n.get("noteId", n.get("note", 0))): n for n in notes}

    reviews_raw = client.get_reviews_of_cards(card_ids)
    cards: list[dict] = []
    all_times: list[int] = []
    total_again = 0
    suspended = 0
    leeches = 0

    for card_id in card_ids:
        info = info_by_card.get(card_id, {})
        note_id = int(info.get("note", info.get("noteId", 0)) or 0)
        note = note_by_id.get(note_id, {})
        tags = list(dict.fromkeys((note.get("tags") or []) + (info.get("tags") or [])))
        rows = _review_rows(reviews_raw, card_id)
        eases = [_review_ease(row) for row in rows]
        again_count = sum(1 for ease in eases if ease == 1)
        times = [t for t in (_review_time_ms(row) for row in rows) if t is not None]
        all_times.extend(times)
        total_again += again_count
        lapses = int(info.get("lapses", 0) or 0)
        is_suspended = int(info.get("queue", 0) or 0) < 0
        is_leech = _is_leech(tags, lapses, min_lapses)
        suspended += int(is_suspended)
        leeches += int(is_leech)
        fields = note.get("fields") or {}
        front_raw = _first_field_value(
            fields, ("Front", "Anverso", "Text", "Pregunta"), fallback_index=0
        ) or str(info.get("question", ""))
        back_raw = _first_field_value(
            fields, ("Back", "Reverso", "Back Extra", "Respuesta"), fallback_index=1
        ) or str(info.get("answer", ""))
        source_raw = _first_field_value(
            fields, ("Source", "Fuente", "Origen"), fallback_index=-1
        )
        front = strip_html_for_display(front_raw)
        back = strip_html_for_display(back_raw)
        source_value = strip_html_for_display(source_raw)
        cards.append({
            "card_id": card_id,
            "note_id": note_id or None,
            "deck": info.get("deckName"),
            "front_excerpt": front[:160],
            "front": front[:160],  # compatibilidad con el contrato inicial
            "lapses": lapses,
            "again_count": again_count,
            "review_count": len(rows),
            "avg_review_time_ms": round(mean(times)) if times else None,
            "interval": info.get("interval"),
            "due": info.get("due"),
            "suspended": is_suspended,
            "leech": is_leech,
            "tags": tags,
            "source": source_value or "anki",
            "origin_source": source_value or "anki",
            "front_full": front,
            "back_full": back,
            "note_type": note.get("modelName") or info.get("modelName") or "Basic",
        })

    ranked = sorted(
        cards,
        key=lambda c: (c["leech"], c["again_count"], c["avg_review_time_ms"] or 0, c["lapses"]),
        reverse=True,
    )
    # Importante: no mutar `ranked`; `_raw_cards` conserva front_full/back_full
    # limpios para embeddings/clustering. La salida pública usa copias compactas.
    compact_cards = [dict(card) for card in ranked[:limit]] if include_cards else []
    for card in compact_cards:
        card.pop("front_full", None)
        card.pop("back_full", None)

    return {
        "query": search,
        "cards_scanned": len(card_ids),
        "notes_scanned": len(note_ids),
        "summary": {
            "leeches": leeches,
            "suspended": suspended,
            "again_count": total_again,
            "avg_review_time_ms": round(mean(all_times)) if all_times else None,
        },
        "cards": compact_cards,
        "_raw_cards": ranked,
    }


def _tag_parts(tags: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for tag in tags:
        if "::" not in tag:
            continue
        category, value = tag.split("::", 1)
        if category and value:
            out[category] = value
    return out


def retention_by_tag(stats: dict, *, threshold: float = 0.80, min_reviews: int = 1, limit: int = 20) -> dict:
    groups: dict[str, dict[str, Any]] = defaultdict(lambda: {"reviews": 0, "again": 0, "time": [], "cards": set()})
    raw_cards = stats.get("_raw_cards") or stats.get("cards") or []
    for card in raw_cards:
        reviews = int(card.get("review_count") or 0)
        if reviews < min_reviews:
            continue
        again = int(card.get("again_count") or 0)
        avg_time = card.get("avg_review_time_ms")
        tags = list(card.get("tags") or [])
        axes = _tag_parts(tags)
        group_keys: list[str] = []
        for category in ("vendor", "cert", "topic", "type"):
            if category in axes:
                group_keys.append(f"{category}::{axes[category]}")
        combo = [axes.get(c) for c in ("vendor", "topic", "type")]
        if all(combo):
            group_keys.append("vendor_topic_type::" + "::".join(combo))
        for key in group_keys:
            g = groups[key]
            g["reviews"] += reviews
            g["again"] += again
            if avg_time is not None:
                g["time"].append(avg_time)
            g["cards"].add(card.get("card_id"))

    rows: list[dict] = []
    for key, g in groups.items():
        reviews = g["reviews"]
        if reviews <= 0:
            continue
        retention = max(0.0, 1.0 - (g["again"] / reviews))
        rows.append({
            "group": key,
            "retention": round(retention, 3),
            "reviews": reviews,
            "again": g["again"],
            "cards": len(g["cards"]),
            "avg_review_time_ms": round(mean(g["time"])) if g["time"] else None,
            "below_threshold": retention < threshold,
        })
    rows.sort(key=lambda r: (r["below_threshold"], -r["again"], -(r["avg_review_time_ms"] or 0)), reverse=True)
    return {"threshold": threshold, "groups": rows[:limit], "summary": {"groups": len(rows), "below_threshold": sum(1 for r in rows if r["below_threshold"])}}


def problematic_records_from_stats(
    stats: dict,
    *,
    settings: Settings,
    taxonomy: Taxonomy,
    profile_name: str,
    profile: ProfileConfig,
) -> list[DuplicateRecord]:
    records: list[DuplicateRecord] = []
    for card in stats.get("_raw_cards") or stats.get("cards") or []:
        if not (card.get("leech") or int(card.get("again_count") or 0) > 0 or (card.get("avg_review_time_ms") or 0) >= 20000):
            continue
        front = strip_html_for_display(card.get("front_full") or card.get("front") or "")
        back = strip_html_for_display(card.get("back_full") or "")
        fp = make_fingerprint(normalize_text(front), normalize_text(back))
        classified = classify_fields(
            front=front,
            back=back,
            source="anki",
            suggested_tags=card.get("tags") or [],
            note_type=card.get("note_type") or "Basic",
            taxonomy=taxonomy,
            fingerprint=fp,
            profile_name=profile_name,
            profile=profile,
            deck=card.get("deck"),
        )
        record = build_record_from_fields(
            candidate_id=f"card:{card.get('card_id')}",
            source="anki",
            origin_source=card.get("origin_source") or "review_stats",
            front=front,
            back=back,
            scope=classified.scope,
            note_type=classified.note_type,
            deck=card.get("deck"),
            anki_note_id=card.get("note_id"),
        )
        classified_type = classified.scope.get("type")
        if record.features.intent == "unknown" and classified_type:
            record = replace(
                record,
                features=replace(record.features, intent=classified_type),
            )
        records.append(record)
    return records


def _serialize_problem_record(record: DuplicateRecord, *, excerpt_chars: int = 160) -> dict:
    return {
        "id": record.candidate_id,
        "source": record.source,
        "origin_source": record.origin_source,
        "deck": record.deck,
        "note_type": record.note_type,
        "front_excerpt": _excerpt(record.front, excerpt_chars),
        "back_excerpt": _excerpt(record.back, excerpt_chars),
        "scope": record.scope.summary(),
        "intent": record.features.intent,
        "semantic_key": record.features.semantic_key,
    }


def _serialize_problem_cluster(
    cluster: DuplicateCluster,
    *,
    max_members: int = 5,
    excerpt_chars: int = 160,
) -> dict:
    members = list(cluster.members)
    compact_members = members[:max_members]
    return {
        "cluster_id": cluster.cluster_id,
        "representative": _serialize_problem_record(
            cluster.representative, excerpt_chars=excerpt_chars
        ),
        "members": [
            _serialize_problem_record(member, excerpt_chars=excerpt_chars)
            for member in compact_members
        ],
        "members_total": len(members),
        "members_returned": len(compact_members),
        "members_truncated": max(0, len(members) - len(compact_members)),
        "reason_codes": list(cluster.reason_codes),
        "score_floor": cluster.score_floor,
    }


def leech_clusters(
    *,
    stats: dict,
    registry: Registry,
    settings: Settings,
    taxonomy: Taxonomy,
    profile_name: str,
    profile: ProfileConfig,
    limit: int = 10,
    max_members: int = 5,
) -> dict:
    records = problematic_records_from_stats(
        stats, settings=settings, taxonomy=taxonomy, profile_name=profile_name, profile=profile
    )
    embeddings_used = False
    if settings.acm.use_embeddings and records:
        cache_lookup, cache_store = embedding_cache_io(registry, settings.acm.ollama_model)
        records = enrich_records_with_embeddings(
            records,
            ollama_url=settings.acm.ollama_url,
            ollama_model=settings.acm.ollama_model,
            cache_lookup=cache_lookup,
            cache_store=cache_store,
        )
        embeddings_used = any(r.features.embedding for r in records)
    clusters, metrics = audit_duplicate_records(
        records,
        cluster_threshold=settings.acm.cluster_threshold,
        similar_threshold=settings.acm.similar_lookup_threshold,
    )
    serialized = []
    for cluster in clusters[:limit]:
        tag_counts = Counter()
        for member in cluster.members:
            for k, v in member.scope.summary().items():
                tag_counts[f"{k}::{v}"] += 1
        serialized.append({
            **_serialize_problem_cluster(cluster, max_members=max_members),
            "label": tag_counts.most_common(1)[0][0] if tag_counts else "sin_tag",
        })
    return {
        "clusters": serialized,
        "metrics": {
            "cards_scanned": metrics.cards_scanned,
            "clusters_found": metrics.clusters_found,
            "clusters_returned": len(serialized),
            "limit": limit,
            "max_members_per_cluster": max_members,
        },
        "embeddings_used": embeddings_used,
    }


def repair_suggestions(stats: dict, clusters: dict | None = None, *, limit: int = 10) -> dict:
    """Modo repair no-destructivo: propone causa/acción; el agente redacta/aplica vía ingest/sync."""
    suggestions: list[dict] = []
    for card in (stats.get("_raw_cards") or stats.get("cards") or [])[:limit]:
        reasons: list[str] = []
        actions: list[str] = []
        if card.get("leech") or int(card.get("lapses") or 0) >= 8:
            reasons.append("leech: muchos lapses")
            actions.append("reescribir prompt o dividir en cards más atómicas")
        if int(card.get("again_count") or 0) > 0:
            reasons.append("muchas respuestas Again")
            actions.append("agregar prerequisito o tarjeta de contraste")
        if (card.get("avg_review_time_ms") or 0) >= 20000:
            reasons.append("respuesta lenta")
            actions.append("aclarar la pista o reducir carga de memoria")
        if not reasons:
            continue
        suggestions.append({
            "card_id": card.get("card_id"),
            "note_id": card.get("note_id"),
            "front_excerpt": _excerpt(card.get("front_full") or card.get("front") or ""),
            "probable_cause": "; ".join(reasons),
            "suggested_action": "; ".join(dict.fromkeys(actions)),
            "workflow": "Confirmar con Eduardo; aplicar con acm_resolve(correct) o acm_ingest -> acm_sync.",
        })
    return {"suggestions": suggestions, "cluster_context": (clusters or {}).get("clusters", [])[:3], "applies_changes": False}
