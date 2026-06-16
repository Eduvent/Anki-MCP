from __future__ import annotations

from acm.anki.client import AnkiConnectClient
from acm.anki.indexer import fetch_deck_records, records_to_index_entries
from acm.config import ProfileConfig, Settings, Taxonomy
from acm.models import CardScope
from acm.pipeline.similarity import (
    DuplicateCluster,
    DuplicateRecord,
    DuplicateMatch,
    DuplicateMetrics,
    build_record_from_fields,
    build_record_from_index_row,
    build_record_from_registry_row,
    enrich_records_with_embeddings,
    find_similar_records,
    audit_duplicate_records,
    serialize_cluster,
)
from acm.pipeline.taxonomy_suggest import suggest_taxonomy_proposals
from acm.store.registry import Registry, embedding_cache_io


def refresh_deck_index(
    *,
    registry: Registry,
    settings: Settings,
    taxonomy: Taxonomy,
    profile_name: str,
    profile: ProfileConfig,
    anki_client: AnkiConnectClient,
    deck: str,
    include_subdecks: bool,
) -> list[DuplicateRecord]:
    records = fetch_deck_records(
        anki_client,
        deck=deck,
        include_subdecks=include_subdecks,
        taxonomy=taxonomy,
        settings=settings,
        profile_name=profile_name,
        profile=profile,
    )
    registry.replace_indexed_notes(
        deck_name=deck,
        include_subdecks=include_subdecks,
        notes=records_to_index_entries(records),
    )
    return records


def _registry_pool(registry: Registry) -> list[DuplicateRecord]:
    # H3/H4: excluye descartada y borrada-en-anki del pool de dedup.
    return [build_record_from_registry_row(row) for row in registry.list_active_cards()]


def _indexed_pool(
    registry: Registry,
    *,
    deck: str,
    include_subdecks: bool,
) -> list[DuplicateRecord]:
    return [
        build_record_from_index_row(row)
        for row in registry.list_indexed_notes(
            deck_name=deck,
            include_subdecks=include_subdecks,
        )
    ]


def build_duplicate_pool(
    *,
    registry: Registry,
    settings: Settings,
    taxonomy: Taxonomy,
    profile_name: str,
    profile: ProfileConfig,
    deck: str | None,
    include_subdecks: bool,
    include_registry: bool,
    anki_client: AnkiConnectClient | None = None,
    refresh_index: bool = False,
    persist_index: bool = False,
) -> list[DuplicateRecord]:
    pool = _registry_pool(registry) if include_registry else []
    anki_available = anki_client is not None

    if deck:
        deck_records = _indexed_pool(
            registry,
            deck=deck,
            include_subdecks=include_subdecks,
        )
        if refresh_index and anki_available:
            deck_records = refresh_deck_index(
                registry=registry,
                settings=settings,
                taxonomy=taxonomy,
                profile_name=profile_name,
                profile=profile,
                anki_client=anki_client,
                deck=deck,
                include_subdecks=include_subdecks,
            )
        elif not deck_records and anki_available:
            live_records = fetch_deck_records(
                anki_client,
                deck=deck,
                include_subdecks=include_subdecks,
                taxonomy=taxonomy,
                settings=settings,
                profile_name=profile_name,
                profile=profile,
            )
            deck_records = live_records
            if persist_index:
                registry.replace_indexed_notes(
                    deck_name=deck,
                    include_subdecks=include_subdecks,
                    notes=records_to_index_entries(live_records),
                )

        pool.extend(deck_records)

    unique = {record.candidate_id: record for record in pool}
    return list(unique.values())


def find_duplicate_clusters(
    *,
    registry: Registry,
    settings: Settings,
    taxonomy: Taxonomy,
    profile_name: str,
    profile: ProfileConfig,
    deck: str,
    include_subdecks: bool,
    include_registry: bool,
    anki_client: AnkiConnectClient | None = None,
    refresh_index: bool = False,
) -> tuple[list[DuplicateCluster], DuplicateMetrics]:
    pool = build_duplicate_pool(
        registry=registry,
        settings=settings,
        taxonomy=taxonomy,
        profile_name=profile_name,
        profile=profile,
        deck=deck,
        include_subdecks=include_subdecks,
        include_registry=include_registry,
        anki_client=anki_client,
        refresh_index=refresh_index,
        persist_index=refresh_index,
    )
    if settings.acm.use_embeddings:
        cache_lookup, cache_store = embedding_cache_io(registry, settings.acm.ollama_model)
        pool = enrich_records_with_embeddings(
            pool,
            ollama_url=settings.acm.ollama_url,
            ollama_model=settings.acm.ollama_model,
            cache_lookup=cache_lookup,
            cache_store=cache_store,
        )
    return audit_duplicate_records(
        pool,
        cluster_threshold=settings.acm.cluster_threshold,
        similar_threshold=settings.acm.similar_lookup_threshold,
    )


def collection_health(
    *,
    registry: Registry,
    settings: Settings,
    taxonomy: Taxonomy,
    profile_name: str,
    profile: ProfileConfig,
    deck: str | None,
    include_subdecks: bool,
    anki_client: AnkiConnectClient | None = None,
) -> dict:
    """E7-2: reporte read-only de salud de la colección — dups + huérfanas + leeches."""
    pool = build_duplicate_pool(
        registry=registry, settings=settings, taxonomy=taxonomy,
        profile_name=profile_name, profile=profile, deck=deck,
        include_subdecks=include_subdecks, include_registry=True,
        anki_client=anki_client, refresh_index=False, persist_index=False,
    )
    if settings.acm.use_embeddings:
        cache_lookup, cache_store = embedding_cache_io(registry, settings.acm.ollama_model)
        pool = enrich_records_with_embeddings(
            pool, ollama_url=settings.acm.ollama_url, ollama_model=settings.acm.ollama_model,
            cache_lookup=cache_lookup, cache_store=cache_store,
        )

    clusters, metrics = audit_duplicate_records(
        pool, cluster_threshold=settings.acm.cluster_threshold,
        similar_threshold=settings.acm.similar_lookup_threshold,
    )

    required = list(dict.fromkeys(
        (profile.required_categories or []) + (profile.routing_categories or [])
    ))
    untagged = sum(
        1 for record in pool
        if any(category not in record.scope.summary() for category in required)
    )

    leeches = 0
    if anki_client is not None and deck:
        for deck_name in anki_client.expand_decks(deck, include_subdecks=include_subdecks):
            leeches += len(anki_client.find_notes(f'deck:"{deck_name}" tag:leech'))

    return {
        "cards_scanned": metrics.cards_scanned,
        "duplicate_clusters": len(clusters),
        "untagged": untagged,
        "leeches": leeches,
        "clusters": [serialize_cluster(cluster) for cluster in clusters[:10]],
    }


def suggest_taxonomy_for_deck(
    *,
    registry: Registry,
    settings: Settings,
    taxonomy: Taxonomy,
    profile_name: str,
    profile: ProfileConfig,
    deck: str | None,
    include_subdecks: bool,
    anki_client: AnkiConnectClient | None = None,
) -> list[dict]:
    """E2-6: arma el pool, lo embebe y propone tags para clusters sin clasificar."""
    pool = build_duplicate_pool(
        registry=registry,
        settings=settings,
        taxonomy=taxonomy,
        profile_name=profile_name,
        profile=profile,
        deck=deck,
        include_subdecks=include_subdecks,
        include_registry=True,
        anki_client=anki_client,
        refresh_index=False,
        persist_index=False,
    )
    if settings.acm.use_embeddings:
        cache_lookup, cache_store = embedding_cache_io(registry, settings.acm.ollama_model)
        pool = enrich_records_with_embeddings(
            pool,
            ollama_url=settings.acm.ollama_url,
            ollama_model=settings.acm.ollama_model,
            cache_lookup=cache_lookup,
            cache_store=cache_store,
        )
    return suggest_taxonomy_proposals(
        pool,
        taxonomy=taxonomy,
        profile=profile,
        cluster_threshold=settings.acm.cluster_threshold,
        similar_threshold=settings.acm.similar_lookup_threshold,
    )


def find_similar_card(
    *,
    registry: Registry,
    settings: Settings,
    taxonomy: Taxonomy,
    profile_name: str,
    profile: ProfileConfig,
    front: str,
    back: str,
    note_type: str,
    deck: str | None,
    include_subdecks: bool,
    anki_client: AnkiConnectClient | None = None,
) -> list[DuplicateMatch]:
    pool = build_duplicate_pool(
        registry=registry,
        settings=settings,
        taxonomy=taxonomy,
        profile_name=profile_name,
        profile=profile,
        deck=deck,
        include_subdecks=include_subdecks,
        include_registry=True,
        anki_client=anki_client,
        refresh_index=False,
        persist_index=False,
    )
    query = build_record_from_fields(
        candidate_id="query",
        source="input",
        origin_source="query",
        front=front,
        back=back,
        scope=CardScope(),
        note_type=note_type,
        deck=deck,
    )
    if settings.acm.use_embeddings:
        cache_lookup, cache_store = embedding_cache_io(registry, settings.acm.ollama_model)
        enriched = enrich_records_with_embeddings(
            [query] + pool,
            ollama_url=settings.acm.ollama_url,
            ollama_model=settings.acm.ollama_model,
            cache_lookup=cache_lookup,
            cache_store=cache_store,
        )
        query = enriched[0]
        pool = enriched[1:]
    return find_similar_records(
        query,
        pool,
        similar_threshold=settings.acm.similar_lookup_threshold,
    )
