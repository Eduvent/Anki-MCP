import pytest

from acm.pipeline.normalizer import build_semantic_key, detect_intent, normalize_text


def test_strips_whitespace():
    assert normalize_text("  hello  ") == "hello"


def test_lowercases():
    assert normalize_text("Azure AD") == "azure ad"


def test_collapses_multiple_spaces():
    assert normalize_text("foo   bar\t\tbaz") == "foo bar baz"


def test_removes_html_tags():
    assert normalize_text("<b>Azure</b> AD") == "azure ad"


def test_unicode_nfkc():
    # Full-width characters → ASCII
    assert normalize_text("Ａｚｕｒｅ") == "azure"


def test_newlines_collapsed():
    assert normalize_text("Azure\nAD\nService") == "azure ad service"


def test_empty_string():
    assert normalize_text("") == ""


def test_already_normalized():
    assert normalize_text("azure ad") == "azure ad"


def test_definition_paraphrases_share_semantic_key():
    key_1 = build_semantic_key("Que es un CDN?")
    key_2 = build_semantic_key("Dime la definicion de un CDN")
    assert key_1 == "definition::cdn"
    assert key_1 == key_2


def test_non_definition_text_has_no_semantic_key():
    assert build_semantic_key("Lista tres ventajas de usar CDN") is None


def test_comparison_key_is_symmetric():
    key_1 = build_semantic_key("Azure CDN vs CloudFront")
    key_2 = build_semantic_key("difference between CloudFront and Azure CDN")
    assert key_1 == "comparison::azure cdn::cloudfront"
    assert key_1 == key_2


def test_detect_intent_defaults_to_unknown():
    assert detect_intent("Lista tres ventajas de usar CDN") == "unknown"


def test_acronym_intent_and_semantic_key():
    assert build_semantic_key("¿Qué significan las siglas EC2?") == "acronym::ec2"
    assert build_semantic_key("What does IAM stand for?") == "acronym::iam"
    assert detect_intent("What does IAM stand for?") == "acronym"


def test_acronym_cross_lingual_shares_key():
    assert build_semantic_key("What does S3 stand for?") == build_semantic_key(
        "¿Qué significan las siglas S3?"
    )


def test_plain_que_significa_stays_definition():
    # sin "siglas" → definición, no acrónimo
    assert build_semantic_key("¿Qué significa un CDN?") == "definition::cdn"
