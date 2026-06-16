import dataclasses
import sqlite3

from acm.models import CardScope
from acm.pipeline.similarity import (
    DuplicateRecord,
    TokenStats,
    _anchor_tokens,
    audit_duplicate_records,
    build_record_from_index_row,
    build_record_from_fields,
    compare_records,
    find_similar_records,
)


def _with_embedding(record: DuplicateRecord, vector: list[float]) -> DuplicateRecord:
    feats = dataclasses.replace(record.features, embedding=tuple(vector))
    return dataclasses.replace(record, features=feats)


def _record(
    candidate_id: str,
    front: str,
    back: str = "",
    *,
    deck: str | None = None,
) -> DuplicateRecord:
    return build_record_from_fields(
        candidate_id=candidate_id,
        source="registry",
        origin_source="manual",
        front=front,
        back=back,
        scope=CardScope(),
        note_type="Basic",
        deck=deck,
    )


def test_definition_paraphrases_cluster_together():
    records = [
        _record("a", "What is a CDN?", "Content Delivery Network", deck="Networking"),
        _record("b", "Que es un CDN", "Red de entrega de contenido", deck="Networking"),
        _record("c", "Dime la definicion de un CDN", "Definicion de CDN", deck="Networking"),
    ]

    clusters, metrics = audit_duplicate_records(
        records,
        cluster_threshold=0.90,
        similar_threshold=0.75,
    )

    assert len(clusters) == 1
    assert {member.candidate_id for member in clusters[0].members} == {"a", "b", "c"}
    assert metrics.cards_scanned == 3


def test_definition_and_non_definition_do_not_cluster():
    records = [
        _record("a", "What is a CDN?", "Content Delivery Network", deck="Networking"),
        _record("b", "Ventajas de usar un CDN", "Reduce latencia", deck="Networking"),
    ]

    clusters, _ = audit_duplicate_records(
        records,
        cluster_threshold=0.90,
        similar_threshold=0.75,
    )

    assert clusters == []


def test_unknown_intent_still_returns_lexical_match():
    candidate = _record("a", "Azure storage account type", deck="Azure")
    query = _record("query", "Azure storage account types", deck="Azure")

    matches = find_similar_records(query, [candidate], similar_threshold=0.75)

    assert matches
    assert matches[0].record.candidate_id == "a"
    assert matches[0].score >= 0.75


def test_medium_edges_do_not_chain_clusters():
    """Cards with moderate lexical overlap but no semantic key shouldn't bridge clusters."""
    strong_a = _record("a", "What is a CDN?", deck="Networking")
    strong_b = _record("b", "Dime la definicion de un CDN", deck="Networking")
    # c shares some tokens with b but has different intent and no semantic key
    medium_c = _record("c", "CDN implementation guide and tutorial", deck="Networking")

    # c should not score >= 0.90 against b (different intent, no semantic key match)
    edge = compare_records(strong_b, medium_c, similar_threshold=0.50)
    assert edge is None or edge.score < 0.90

    clusters, _ = audit_duplicate_records(
        [strong_a, strong_b, medium_c],
        cluster_threshold=0.90,
        similar_threshold=0.75,
    )

    # a and b cluster together via semantic key; c stays out
    definition_clusters = [c for c in clusters if {m.candidate_id for m in c.members} >= {"a", "b"}]
    assert len(definition_clusters) == 1
    assert "c" not in {m.candidate_id for m in definition_clusters[0].members}


def test_embeddings_do_not_drag_strong_lexical_below_threshold():
    """E0-2: par léxico fuerte (Jaccard=1.0 → 0.90) con coseno 0.80 debe seguir
    clusterizando. Antes el coseno pisaba la léxica y el par caía a 0.80 < 0.90."""
    a = _record("a", "What are the storage account replication options in azure", deck="Azure")
    b = _record("b", "azure storage account replication options what are the", deck="Azure")
    # cosine([1,0],[0.8,0.6]) = 0.8
    a = _with_embedding(a, [1.0, 0.0])
    b = _with_embedding(b, [0.8, 0.6])

    edge = compare_records(a, b, similar_threshold=0.75)
    assert edge is not None
    assert edge.score >= 0.90  # léxica fuerte preservada, no pisada por coseno 0.80

    clusters, _ = audit_duplicate_records(
        [a, b], cluster_threshold=0.90, similar_threshold=0.75
    )
    assert len(clusters) == 1
    assert {m.candidate_id for m in clusters[0].members} == {"a", "b"}


def test_embeddings_raise_recall_for_paraphrase_without_lexical_overlap():
    """E0-2/E1: coseno alto sube el score aunque la léxica sea débil."""
    a = _record("a", "Define horizontal scaling", deck="Arch")
    b = _record("b", "Define horizontal scaling", deck="Arch")  # same block keys
    a = _with_embedding(a, [1.0, 0.0])
    b = _with_embedding(b, [0.95, 0.31])  # cosine ~0.95

    edge = compare_records(a, b, similar_threshold=0.75)
    assert edge is not None
    assert "embedding_match" in edge.reason_codes


def test_idf_prefers_rare_tokens_as_anchors():
    """IDF-weighted anchor selection should prefer rare tokens over common ones."""
    stats = TokenStats()
    # "azure" appears in many documents; "bicep" appears in one
    common_tokens = ("azure", "storage", "account", "bicep")
    for _ in range(10):
        stats.update(("azure", "storage", "account"))
    stats.update(("bicep",))

    anchors = _anchor_tokens(common_tokens, stats)
    # "bicep" should rank first because it's the rarest
    assert anchors[0] == "bicep"


def test_build_record_from_index_row_accepts_sqlite_row():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE indexed_notes (
            note_id INTEGER,
            deck_name TEXT,
            note_type TEXT,
            front_original TEXT,
            back_original TEXT,
            front_normalized TEXT,
            back_normalized TEXT,
            fingerprint TEXT,
            scope_vendor TEXT,
            scope_topic TEXT,
            scope_cert TEXT,
            scope_json TEXT,
            content_tokens TEXT,
            back_tokens TEXT,
            char_trigrams TEXT,
            anchor_tokens TEXT,
            block_keys TEXT,
            intent TEXT,
            semantic_key TEXT,
            trigram_signature TEXT
        )
        """
    )
    conn.execute(
        """
        INSERT INTO indexed_notes VALUES (
            123,
            'Cloud Certs::AWS::AWS Cloud Practitioner',
            'Basic',
            'What is AWS CAF?',
            'A framework for cloud adoption.',
            'what is aws caf',
            'a framework for cloud adoption',
            'fp-123',
            'aws',
            NULL,
            'aws-cloud-practitioner',
            '{"vendor":"aws","cert":"aws-cloud-practitioner"}',
            'aws caf framework',
            'framework cloud adoption',
            'wha|hat',
            'aws caf',
            'deck:cloud-certs|semantic:definition::aws-caf',
            'definition',
            'definition::aws caf',
            'wha|hat'
        )
        """
    )
    row = conn.execute("SELECT * FROM indexed_notes").fetchone()

    record = build_record_from_index_row(row)

    assert record.candidate_id == "anki:123"
    assert record.features.intent == "definition"
    assert record.deck == "Cloud Certs::AWS::AWS Cloud Practitioner"
    conn.close()
