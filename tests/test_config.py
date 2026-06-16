from acm.config import Settings


def test_duplicate_threshold_backfills_cluster_threshold():
    settings = Settings(acm={"duplicate_threshold": 0.93})
    assert settings.acm.cluster_threshold == 0.93
    assert settings.acm.similar_lookup_threshold == 0.75
