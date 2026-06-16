from __future__ import annotations

from acm.anki.client import AnkiConnectClient
from acm.anki.indexer import fetch_deck_records
from acm.config import ProfileConfig, Settings, Taxonomy, load_profile_taxonomy
from acm.models import AuditDecision, CandidateCard
from acm.pipeline.classifier import classify
from acm.pipeline.llm import classify_with_llm
from acm.pipeline.normalizer import normalize_text
from acm.pipeline.propagation import (
    apply_high_confidence_propagation,
    propagate_facets_from_neighbors,
)
from acm.pipeline.similarity import (
    DuplicateRecord,
    build_record_from_classified,
    build_record_from_index_row,
    build_record_from_registry_row,
    enrich_records_with_embeddings,
    find_similar_records,
    make_fingerprint,
    serialize_match,
)
from acm.pipeline.quality import suggest_better_version
from acm.store.registry import Registry, embedding_cache_io


def _registry_records(registry: Registry) -> list[DuplicateRecord]:
    return [
        build_record_from_registry_row(row)
        for row in registry.list_processed_cards()
    ]


def _all_indexed_records(registry: Registry) -> list[DuplicateRecord]:
    """E1-5: TODA la colección indexada (todos los mazos), no solo el destino.

    Es lo que habilita la dedup cross-deck (RF-B2): una card nueva se compara
    contra lo que ya existe en cualquier mazo + el registro, no solo su mazo
    ruteado.
    """
    return [
        build_record_from_index_row(row)
        for row in registry.list_indexed_notes(deck_name=None)
    ]


def _cached_or_live_deck_records(
    *,
    registry: Registry,
    taxonomy: Taxonomy,
    profile_name: str,
    profile: ProfileConfig,
    settings: Settings,
    anki_client: AnkiConnectClient | None,
    deck: str,
    include_subdecks: bool,
    deck_cache: dict[tuple[str, bool], list[DuplicateRecord]],
) -> list[DuplicateRecord]:
    cache_key = (deck, include_subdecks)
    if cache_key in deck_cache:
        return deck_cache[cache_key]

    indexed_rows = registry.list_indexed_notes(
        deck_name=deck,
        include_subdecks=include_subdecks,
    )
    if indexed_rows:
        records = [build_record_from_index_row(row) for row in indexed_rows]
        deck_cache[cache_key] = records
        return records

    if anki_client is not None:
        records = fetch_deck_records(
            anki_client,
            deck=deck,
            include_subdecks=include_subdecks,
            taxonomy=taxonomy,
            settings=settings,
            profile_name=profile_name,
            profile=profile,
        )
        deck_cache[cache_key] = records
        return records

    deck_cache[cache_key] = []
    return []


def _format_match_reason(score: float, reason_codes: tuple[str, ...]) -> str:
    codes = ", ".join(reason_codes)
    return f"Coincidencia fuerte detectada ({codes}, score={score:.2f})"


def _resolve_classification_context(
    *,
    card: CandidateCard,
    settings: Settings,
    taxonomy: Taxonomy,
    profile_name: str | None,
    profile: ProfileConfig | None,
    context_cache: dict[str, tuple[str, ProfileConfig, Taxonomy]] | None,
) -> tuple[str, ProfileConfig, Taxonomy]:
    requested_profile = card.profile or profile_name or settings.default_profile_name
    if context_cache is not None and requested_profile in context_cache:
        return context_cache[requested_profile]

    if requested_profile == (profile_name or settings.default_profile_name):
        resolved_name = requested_profile
        resolved_profile = profile or settings.get_profile(requested_profile)[1]
        context = (resolved_name, resolved_profile, taxonomy)
    else:
        context = load_profile_taxonomy(settings, requested_profile)

    if context_cache is not None:
        context_cache[requested_profile] = context
    return context


def audit_card(
    card: CandidateCard,
    registry: Registry,
    taxonomy: Taxonomy,
    settings: Settings,
    anki_client: AnkiConnectClient | None = None,
    *,
    profile_name: str | None = None,
    profile: ProfileConfig | None = None,
    context_cache: dict[str, tuple[str, ProfileConfig, Taxonomy]] | None = None,
    registry_records: list[DuplicateRecord] | None = None,
    indexed_records: list[DuplicateRecord] | None = None,
    batch_records: list[DuplicateRecord] | None = None,
    deck_cache: dict[tuple[str, bool], list[DuplicateRecord]] | None = None,
) -> AuditDecision:
    """Orquesta el pipeline completo para una sola tarjeta."""
    resolved_profile_name, resolved_profile, resolved_taxonomy = _resolve_classification_context(
        card=card,
        settings=settings,
        taxonomy=taxonomy,
        profile_name=profile_name,
        profile=profile,
        context_cache=context_cache,
    )
    fingerprint = make_fingerprint(
        normalize_text(card.front),
        normalize_text(card.back),
    )
    classified = classify(
        card,
        resolved_taxonomy,
        fingerprint,
        profile_name=resolved_profile_name,
        profile=resolved_profile,
    )
    query_record = build_record_from_classified(
        classified,
        candidate_id="query",
        source="input",
        origin_source=card.source,
        deck=classified.deck,
    )

    candidate_records: list[DuplicateRecord] = []
    candidate_records.extend(registry_records if registry_records is not None else _registry_records(registry))
    # E1-5: dedup cross-deck — incluir TODA la colección indexada, no solo el
    # mazo destino. El live-fetch del mazo ruteado (abajo) complementa con notas
    # recién creadas que todavía no están en el índice.
    candidate_records.extend(indexed_records if indexed_records is not None else _all_indexed_records(registry))
    if batch_records:
        candidate_records.extend(batch_records)

    include_subdecks = settings.acm.audit_include_subdecks
    cache = deck_cache if deck_cache is not None else {}
    deck = classified.deck
    if deck is None and anki_client is not None and resolved_profile.root_deck:
        deck = anki_client.resolve_deck(
            scope=classified.scope,
            root_deck=resolved_profile.root_deck,
            routing_categories=resolved_profile.routing_categories,
        )
    if anki_client is not None and deck:
        candidate_records.extend(
            _cached_or_live_deck_records(
                registry=registry,
                taxonomy=resolved_taxonomy,
                profile_name=resolved_profile_name,
                profile=resolved_profile,
                settings=settings,
                anki_client=anki_client,
                deck=deck,
                include_subdecks=include_subdecks,
                deck_cache=cache,
            )
        )

    # El motor trabaja sobre IDs únicos; evitamos comparar duplicados del pool.
    unique_candidates = {
        record.candidate_id: record
        for record in candidate_records
        if record.candidate_id != query_record.candidate_id
    }

    candidates_list = list(unique_candidates.values())
    if settings.acm.use_embeddings:
        cache_lookup, cache_store = embedding_cache_io(registry, settings.acm.ollama_model)
        enriched = enrich_records_with_embeddings(
            [query_record] + candidates_list,
            ollama_url=settings.acm.ollama_url,
            ollama_model=settings.acm.ollama_model,
            cache_lookup=cache_lookup,
            cache_store=cache_store,
        )
        query_record = enriched[0]
        candidates_list = enriched[1:]

    matches = find_similar_records(
        query_record,
        candidates_list,
        similar_threshold=settings.acm.similar_lookup_threshold,
    )

    # E2-3: propagación de tags por kNN. Para las categorías que el clasificador
    # determinista no resolvió, heredar la faceta más votada entre los vecinos ya
    # etiquetados (alta confianza → auto-aplica; lo demás queda para el usuario).
    propagated = propagate_facets_from_neighbors(matches, taxonomy=resolved_taxonomy)
    apply_high_confidence_propagation(classified, propagated)

    # E2-4/E2-5: fallback con LLM local para las categorías requeridas que ni el
    # determinista ni el kNN resolvieron. Es el escalón previo a Claude — solo el
    # residuo que tampoco resuelve el LLM local queda como ambiguo para el agente.
    if settings.acm.use_llm_fallback:
        missing = [
            category
            for category in resolved_profile.required_categories
            if classified.scope.get(category) is None
        ]
        if missing:
            llm_facets = classify_with_llm(
                classified.front,
                classified.back,
                resolved_taxonomy,
                ollama_url=settings.acm.ollama_url,
                model=settings.acm.ollama_chat_model,
                categories=missing,
            )
            for category, value in llm_facets.items():
                if classified.scope.get(category) is None:
                    classified.scope.set(category, value)
                    tag = f"{category}::{value}"
                    if tag not in classified.tags_resolved:
                        classified.tags_resolved.append(tag)

    strong_matches = [
        match
        for match in matches
        if match.score >= settings.acm.cluster_threshold
    ]
    if strong_matches:
        best = strong_matches[0]
        return AuditDecision(
            card=classified,
            action="possible_duplicate",
            reason=_format_match_reason(best.score, best.reason_codes),
            matches=[match.record.candidate_id for match in strong_matches],
            match_details=[
                {
                    **serialize_match(match),
                    # E8-1: ¿la nueva es mejor versión que la existente?
                    "mejor_version": suggest_better_version(
                        classified.front, classified.back, match.record.front, match.record.back
                    ),
                }
                for match in strong_matches[:5]
            ],
        )

    total_suggested = len(card.suggested_tags)
    if total_suggested > 0:
        unresolved_count = len(classified.tags_unresolved)
        if unresolved_count / total_suggested > 0.5:
            return AuditDecision(
                card=classified,
                action="reject",
                reason=(
                    f"{unresolved_count}/{total_suggested} tags sugeridos no están en la taxonomía (>50%)"
                ),
            )

    return AuditDecision(
        card=classified,
        action="insert",
        reason="Tarjeta válida, sin duplicados fuertes",
    )


def correct_record(
    record_id: str,
    *,
    front: str,
    back: str,
    tags: list[str] | None,
    registry: Registry,
    taxonomy: Taxonomy,
    settings: Settings,
    profile_name: str,
    profile: ProfileConfig,
    anki_client: AnkiConnectClient | None = None,
) -> AuditDecision | None:
    """E5-3: una corrección reingresa al pipeline (re-dedup + re-clasifica).

    Excluye el propio registro del pool (para no auto-matchearse) y reescribe la
    fila con la decisión fresca. Devuelve la decisión, o None si no existe.
    """
    row = registry.get_by_id(record_id)
    if row is None:
        return None

    card = CandidateCard(
        front=front,
        back=back,
        source=row["source"],
        suggested_tags=tags or [],
        note_type=row["note_type"] or "Basic",
        profile=row["profile_name"] if "profile_name" in row.keys() else None,
        deck=row["target_deck"] if "target_deck" in row.keys() else None,
        material_origen=row["material_origen"] if "material_origen" in row.keys() else None,
    )
    # Excluir el registro en corrección del pool para que no se auto-detecte dup.
    registry_records = [
        record for record in _registry_records(registry) if record.candidate_id != record_id
    ]
    decision = audit_card(
        card, registry, taxonomy, settings, anki_client,
        profile_name=profile_name, profile=profile, registry_records=registry_records,
    )
    registry.update_from_decision(record_id, decision)
    return decision


def audit_batch(
    cards: list[CandidateCard],
    registry: Registry,
    taxonomy: Taxonomy,
    settings: Settings,
    anki_client: AnkiConnectClient | None = None,
    *,
    profile_name: str | None = None,
    profile: ProfileConfig | None = None,
) -> list[AuditDecision]:
    """Procesa una lista de tarjetas candidatas con contexto compartido."""
    decisions: list[AuditDecision] = []
    batch_records: list[DuplicateRecord] = []
    registry_records = _registry_records(registry)
    indexed_records = _all_indexed_records(registry)  # E1-5: colección completa, una vez
    deck_cache: dict[tuple[str, bool], list[DuplicateRecord]] = {}
    context_cache: dict[str, tuple[str, ProfileConfig, Taxonomy]] = {}

    for index, card in enumerate(cards):
        decision = audit_card(
            card,
            registry,
            taxonomy,
            settings,
            anki_client,
            profile_name=profile_name,
            profile=profile,
            context_cache=context_cache,
            registry_records=registry_records,
            indexed_records=indexed_records,
            batch_records=batch_records,
            deck_cache=deck_cache,
        )
        decisions.append(decision)
        batch_records.append(
            build_record_from_classified(
                decision.card,
                candidate_id=f"batch:{index}",
                source="input",
                origin_source=decision.card.source,
                deck=decision.card.deck,
            )
        )

    return decisions
