import pytest

from acm.config import ProfileConfig, Taxonomy
from acm.models import CandidateCard
from acm.pipeline.classifier import classify
from acm.pipeline.similarity import make_fingerprint
from acm.pipeline.normalizer import normalize_text


def _fp(front: str, back: str) -> str:
    return make_fingerprint(normalize_text(front), normalize_text(back))


def _taxonomy() -> Taxonomy:
    return Taxonomy(
        vendor=["azure", "aws", "gcp", "general"],
        cert=["az900", "az104", "saa-c03"],
        topic=["identity", "networking", "storage", "compute", "security"],
        type=["definition", "comparison", "command"],
    )


def _profile() -> ProfileConfig:
    """E2-1: reglas de dominio como DATOS (no en el código). El clasificador
    determinista solo detecta facetas si el perfil las define."""
    return ProfileConfig(
        keyword_rules={"vendor": {
            "azure": ["azure", "azure ad", "entra"],
            "aws": ["aws", "ec2", "s3 bucket", "lambda"],
        }},
        topic_keywords={
            "identity": ["identity", "iam", "azure ad", "entra"],
            "networking": ["dns", "vpc", "load balancer"],
        },
        # type_patterns vacío → usa el default genérico (lingüístico) del código.
    )


def test_detects_azure_vendor():
    card = CandidateCard(front="What is Azure AD?", back="Identity service", source="claude")
    fp = _fp(card.front, card.back)
    result = classify(card, _taxonomy(), fp, profile=_profile())
    assert result.scope.vendor == "azure"


def test_detects_aws_vendor():
    card = CandidateCard(front="What is EC2?", back="Virtual machine in AWS", source="claude")
    fp = _fp(card.front, card.back)
    result = classify(card, _taxonomy(), fp, profile=_profile())
    assert result.scope.vendor == "aws"


def test_no_vendor_when_unrecognized():
    card = CandidateCard(front="What is DNS?", back="Domain Name System", source="manual")
    fp = _fp(card.front, card.back)
    result = classify(card, _taxonomy(), fp, profile=_profile())
    assert result.scope.vendor is None


def test_no_keyword_detection_without_profile_rules():
    """E2-1: sin reglas en el perfil, el determinista no inventa facetas de dominio."""
    card = CandidateCard(front="What is Azure AD?", back="Identity service", source="claude")
    fp = _fp(card.front, card.back)
    result = classify(card, _taxonomy(), fp)  # profile=None → sin keywords
    assert result.scope.vendor is None
    assert result.scope.topic is None


def test_resolves_valid_tags():
    card = CandidateCard(
        front="What is Azure AD?",
        back="Identity service",
        source="claude",
        suggested_tags=["vendor::azure", "topic::identity", "type::definition"],
    )
    fp = _fp(card.front, card.back)
    result = classify(card, _taxonomy(), fp)
    assert "vendor::azure" in result.tags_resolved
    assert "topic::identity" in result.tags_resolved
    assert "type::definition" in result.tags_resolved
    assert result.tags_unresolved == []


def test_marks_unresolved_tags():
    card = CandidateCard(
        front="What is Azure AD?",
        back="Identity service",
        source="claude",
        suggested_tags=["vendor::azure", "topic::invented_topic"],
    )
    fp = _fp(card.front, card.back)
    result = classify(card, _taxonomy(), fp)
    assert "vendor::azure" in result.tags_resolved
    assert any("invented_topic" in t for t in result.tags_unresolved)


def test_tags_without_separator_are_unresolved():
    card = CandidateCard(
        front="What is Azure AD?",
        back="Identity service",
        source="claude",
        suggested_tags=["azure-identity", "definition"],
    )
    fp = _fp(card.front, card.back)
    result = classify(card, _taxonomy(), fp, profile=_profile())
    assert len(result.tags_unresolved) == 2
    assert "vendor::azure" in result.tags_resolved
    assert "topic::identity" in result.tags_resolved
    assert "type::definition" in result.tags_resolved


def test_fingerprint_set():
    card = CandidateCard(front="What is Azure AD?", back="Identity service", source="claude")
    fp = _fp(card.front, card.back)
    result = classify(card, _taxonomy(), fp)
    assert result.fingerprint == fp
    assert len(result.fingerprint) == 64  # SHA256 hex
