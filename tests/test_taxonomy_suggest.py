"""E2-6 · taxonomía auto-sugerida desde clusters de cards sin clasificar."""

import dataclasses

from acm.config import ProfileConfig, Taxonomy
from acm.models import CardScope
from acm.pipeline.similarity import build_record_from_fields
from acm.pipeline.taxonomy_suggest import suggest_taxonomy_proposals


def _rec(cid, front, vec, facets=None):
    r = build_record_from_fields(
        candidate_id=cid, source="anki", origin_source="anki",
        front=front, back="", scope=CardScope(facets=facets or {}), note_type="Basic",
    )
    return dataclasses.replace(r, features=dataclasses.replace(r.features, embedding=tuple(vec)))


def test_proposes_label_from_untagged_cluster():
    tax = Taxonomy(topic=["existing"])
    profile = ProfileConfig(required_categories=["topic"])
    pool = [
        _rec("a", "What is a kubernetes pod", [1.0, 0.0]),
        _rec("b", "Explain kubernetes deployment", [0.99, 0.141]),
        _rec("c", "kubernetes service networking", [0.98, 0.2]),
    ]
    proposals = suggest_taxonomy_proposals(
        pool, taxonomy=tax, profile=profile, cluster_threshold=0.9, similar_threshold=0.75,
    )
    assert proposals
    top = proposals[0]
    assert top["suggested_value"] == "kubernetes"
    assert top["suggested_category"] == "topic"
    assert top["is_new"] is True
    assert top["cluster_size"] == 3


def test_tagged_cards_are_not_proposed():
    tax = Taxonomy(topic=["compute"])
    profile = ProfileConfig(required_categories=["topic"])
    # ya tienen topic → no son "sin clasificar"
    pool = [
        _rec("a", "kubernetes pod", [1.0, 0.0], facets={"topic": "compute"}),
        _rec("b", "kubernetes deploy", [0.99, 0.141], facets={"topic": "compute"}),
        _rec("c", "kubernetes svc", [0.98, 0.2], facets={"topic": "compute"}),
    ]
    assert suggest_taxonomy_proposals(
        pool, taxonomy=tax, profile=profile, cluster_threshold=0.9, similar_threshold=0.75,
    ) == []


def test_below_min_cluster_size_no_proposal():
    tax = Taxonomy(topic=[])
    profile = ProfileConfig(required_categories=["topic"])
    pool = [_rec("a", "lonely card here", [1.0, 0.0])]
    assert suggest_taxonomy_proposals(
        pool, taxonomy=tax, profile=profile, cluster_threshold=0.9, similar_threshold=0.75,
    ) == []
