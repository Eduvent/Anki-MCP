import json

from acm.config import load_profile_taxonomy, load_settings
from acm.pipeline.review_stats import retention_by_tag, review_stats
from acm.service import build_note_fields
from acm.store.registry import Registry


class FakeReviewClient:
    def find_cards(self, query):
        assert query
        return [101, 102]

    def cards_info(self, card_ids):
        return [
            {"cardId": 101, "note": 1, "deckName": "Cloud::AWS", "lapses": 8, "queue": 2, "interval": 3, "due": 10},
            {"cardId": 102, "note": 2, "deckName": "Cloud::AWS", "lapses": 1, "queue": -1, "interval": 7, "due": 20},
        ]

    def get_notes_info(self, note_ids):
        return [
            {"noteId": 1, "tags": ["leech", "vendor::aws", "topic::networking", "type::comparison"], "modelName": "Basic", "fields": {"Front": {"value": "NAT vs IGW"}, "Back": {"value": "Contrast"}}},
            {"noteId": 2, "tags": ["vendor::aws", "topic::identity", "type::definition"], "modelName": "Basic", "fields": {"Front": {"value": "IAM"}, "Back": {"value": "Identity"}}},
        ]

    def get_reviews_of_cards(self, card_ids):
        return {"101": [[1, 101, 0, 1, 0, 0, 0, 30000, 1], [2, 101, 0, 3, 0, 0, 0, 10000, 1]], "102": [[3, 102, 0, 1, 0, 0, 0, 5000, 1]]}

    def close(self):
        pass


def test_review_stats_extracts_lapses_again_time_and_status():
    data = review_stats(FakeReviewClient(), deck="Cloud", limit=10)
    assert data["cards_scanned"] == 2
    assert data["summary"]["leeches"] == 1
    assert data["summary"]["suspended"] == 1
    assert data["summary"]["again_count"] == 2
    first = data["cards"][0]
    assert first["leech"] is True
    assert first["again_count"] == 1
    assert first["avg_review_time_ms"] == 20000


def test_retention_by_tag_groups_topic_type_and_combo():
    stats = review_stats(FakeReviewClient(), deck="Cloud", limit=10)
    report = retention_by_tag(stats, threshold=0.8)
    groups = {row["group"]: row for row in report["groups"]}
    assert "topic::networking" in groups
    assert groups["topic::networking"]["retention"] == 0.5
    assert groups["topic::identity"]["below_threshold"] is True
    assert "vendor_topic_type::aws::networking::comparison" in groups


def test_source_field_is_mapped_when_model_has_it():
    fields = build_note_fields(["Text", "Back Extra", "Source"], "Q", "A", source="claude", material_origen="mod1#p2")
    assert fields == {"Text": "Q", "Back Extra": "A", "Source": "mod1#p2"}


def test_acm_decks_contract(monkeypatch):
    import acm.mcp_server as srv

    class FakeDeckClient:
        def get_decks(self):
            return ["Root", "Root::Child"]
        def deck_card_count(self, deck, include_subdecks=True):
            return 3 if deck == "Root" else 1
        def close(self):
            pass

    monkeypatch.setattr(srv, "_try_anki_client", lambda settings: FakeDeckClient())
    result = json.loads(srv.acm_decks())
    assert result["anki_available"] is True
    assert result["decks"][0]["card_count"] == 3


def test_acm_review_stats_contract(monkeypatch):
    import acm.mcp_server as srv
    monkeypatch.setattr(srv, "_try_anki_client", lambda settings: FakeReviewClient())
    result = json.loads(srv.acm_review_stats(deck="Cloud"))
    assert result["embeddings_used"] is False
    assert result["summary"]["leeches"] == 1


def test_acm_repair_is_non_destructive(monkeypatch, tmp_path):
    import acm.mcp_server as srv
    settings = load_settings()
    settings.acm.use_embeddings = False
    name, profile, taxonomy = load_profile_taxonomy(settings)
    reg = Registry(tmp_path / "t.db")
    monkeypatch.setattr(srv, "_setup", lambda *a, **k: (reg, settings, name, profile, taxonomy))
    monkeypatch.setattr(srv, "_try_anki_client", lambda settings: FakeReviewClient())
    result = json.loads(srv.acm_repair(deck="Cloud"))
    assert result["applies_changes"] is False
    assert result["suggestions"]


def test_quality_flags_with_source_marks_missing_source():
    import acm.mcp_server as srv
    flags = srv._quality_flags_with_source("Q", "A", "")
    assert any(flag["code"] == "fuente_faltante" for flag in flags)


def _problem_stats_with_html():
    style = "<style>.card{font-family:Arial;color:red}.x{padding:0}</style>"
    cards = []
    pairs = [
        ("What is NAT Gateway?", "A managed NAT service", ["vendor::aws", "topic::networking", "type::definition"]),
        ("What is IAM?", "Identity and Access Management", ["vendor::aws", "topic::identity", "type::definition"]),
        ("What is S3?", "Object storage", ["vendor::aws", "topic::storage", "type::definition"]),
    ]
    cid = 1
    for front, back, tags in pairs:
        for _ in range(2):
            clean_front = style + front
            cards.append({
                "card_id": cid,
                "note_id": cid,
                "deck": "Cloud::AWS",
                "front": clean_front[:160],
                "front_full": clean_front,
                "back_full": back,
                "lapses": 9,
                "again_count": 2,
                "review_count": 3,
                "avg_review_time_ms": 30000,
                "leech": True,
                "tags": tags,
                "source": "anki",
                "origin_source": "anki",
                "note_type": "Basic",
            })
            cid += 1
    return {"_raw_cards": cards, "cards": cards, "summary": {"leeches": len(cards)}}


def test_leech_clusters_limits_and_strips_front_html(tmp_path):
    from acm.pipeline.review_stats import leech_clusters

    settings = load_settings()
    settings.acm.use_embeddings = False
    name, profile, taxonomy = load_profile_taxonomy(settings)
    reg = Registry(tmp_path / "t.db")

    one = leech_clusters(
        stats=_problem_stats_with_html(), registry=reg, settings=settings,
        taxonomy=taxonomy, profile_name=name, profile=profile, limit=1, max_members=1,
    )
    three = leech_clusters(
        stats=_problem_stats_with_html(), registry=reg, settings=settings,
        taxonomy=taxonomy, profile_name=name, profile=profile, limit=3, max_members=1,
    )

    assert one["metrics"]["clusters_returned"] == 1
    assert three["metrics"]["clusters_returned"] >= 2
    rep = one["clusters"][0]["representative"]
    assert "front_excerpt" in rep
    assert "front" not in rep
    assert "<style" not in rep["front_excerpt"]
    assert ".card" not in rep["front_excerpt"]
    assert rep["intent"] == "definition"
    assert one["clusters"][0]["members_returned"] == 1


def test_acm_periodic_report_uses_same_leech_definition(monkeypatch):
    import acm.mcp_server as srv
    monkeypatch.setattr(srv, "_try_anki_client", lambda settings: FakeReviewClient())
    result = json.loads(srv.acm_periodic_report(deck="Cloud"))
    assert result["summary"]["leeches"] == len(result["leeches"]) == 1


def test_review_stats_strips_rendered_anki_css():
    class HtmlClient(FakeReviewClient):
        def get_notes_info(self, note_ids):
            return [{"noteId": 1, "tags": ["leech"], "modelName": "Basic", "fields": {}}]
        def cards_info(self, card_ids):
            return [{"cardId": 101, "note": 1, "deckName": "Cloud", "lapses": 8, "queue": 2, "question": "<style>.card{font:12px}</style><div>What is NAT?</div>", "answer": "<b>Network address translation</b>"}]
        def find_cards(self, query):
            return [101]
        def get_reviews_of_cards(self, card_ids):
            return {"101": [[1, 101, 0, 1, 0, 0, 0, 1000, 1]]}

    data = review_stats(HtmlClient(), deck="Cloud")
    assert data["cards"][0]["front_excerpt"] == "What is NAT?"
    assert data["_raw_cards"][0]["back_full"] == "Network address translation"
