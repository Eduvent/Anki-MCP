from __future__ import annotations

import re

from acm.config import (
    DEFAULT_PROFILE_TYPE_PATTERNS,
    ProfileConfig,
    Taxonomy,
)
from acm.models import CandidateCard, CardScope, ClassifiedCard
from acm.pipeline.normalizer import detect_intent, normalize_semantic_text, normalize_text


def _parse_tag(tag: str) -> tuple[str, str] | None:
    """Parsea 'category::value' → (category, value). Retorna None si no tiene el formato."""
    if "::" not in tag:
        return None
    parts = tag.split("::", 1)
    category = parts[0].strip()
    value = parts[1].strip()
    if not category or not value:
        return None
    return category, value


def detect_scope_facets(text: str, profile: ProfileConfig | None) -> dict[str, str]:
    """Detecta facetas por keywords del perfil activo (E2-1: data-driven).

    Sin perfil o sin reglas → no detecta nada por keyword (se delega a la
    propagación por kNN / LLM local / tags explícitos). No hay keywords de
    dominio hardcodeadas en el código.
    """
    keyword_rules = profile.keyword_rules if profile and profile.keyword_rules else {}

    normalized = normalize_text(text)
    detected: dict[str, str] = {}
    for category, value_keywords in keyword_rules.items():
        for value, keywords in value_keywords.items():
            if any(keyword in normalized for keyword in keywords):
                detected[category] = value
                break
    return detected


def _normalize_deck_parts(deck: str) -> list[str]:
    return [
        normalize_semantic_text(part)
        for part in deck.split("::")
        if normalize_semantic_text(part)
    ]


def detect_deck_facets(deck: str | None, profile: ProfileConfig | None) -> dict[str, str]:
    """Resuelve facetas determinísticas a partir del path completo del deck."""
    if not deck:
        return {}

    deck_tag_rules = profile.deck_tag_rules if profile and profile.deck_tag_rules else {}
    if not deck_tag_rules:
        return {}

    deck_parts = _normalize_deck_parts(deck)
    if not deck_parts:
        return {}

    for configured_path, facets in deck_tag_rules.items():
        configured_parts = _normalize_deck_parts(configured_path)
        if not configured_parts:
            continue
        if deck_parts == configured_parts or deck_parts[-len(configured_parts):] == configured_parts:
            return {
                category: value.strip()
                for category, value in facets.items()
                if isinstance(value, str) and value.strip()
            }
    return {}


def detect_topic_facet(
    text: str,
    taxonomy: Taxonomy,
    profile: ProfileConfig | None,
) -> str | None:
    topic_keywords = profile.topic_keywords if profile and profile.topic_keywords else {}
    if not topic_keywords:
        return None

    normalized = normalize_semantic_text(text)
    valid_topics = set(taxonomy.values_for("topic"))
    for topic, keywords in topic_keywords.items():
        if valid_topics and topic not in valid_topics:
            continue
        for keyword in keywords:
            if normalize_semantic_text(keyword) in normalized:
                return topic
    return None


def detect_type_facet(
    front: str,
    taxonomy: Taxonomy,
    profile: ProfileConfig | None,
) -> str | None:
    type_patterns = profile.type_patterns if profile and profile.type_patterns else DEFAULT_PROFILE_TYPE_PATTERNS
    if not type_patterns:
        return None

    normalized_front = normalize_semantic_text(front)
    valid_types = set(taxonomy.values_for("type"))

    for type_name, patterns in type_patterns.items():
        if valid_types and type_name not in valid_types:
            continue
        if any(re.search(pattern, normalized_front, flags=re.IGNORECASE) for pattern in patterns):
            return type_name

    detected_intent = detect_intent(front)
    if detected_intent in valid_types:
        return detected_intent

    return None


def build_scope(
    *,
    front: str,
    combined_text: str,
    suggested_tags: list[str],
    taxonomy: Taxonomy,
    profile: ProfileConfig | None,
    deck: str | None = None,
) -> tuple[CardScope, list[str], list[str]]:
    detected_facets = detect_deck_facets(deck, profile)
    for category, value in detect_scope_facets(combined_text, profile).items():
        detected_facets.setdefault(category, value)

    detected_topic = detect_topic_facet(combined_text, taxonomy, profile)
    if detected_topic:
        detected_facets.setdefault("topic", detected_topic)

    detected_type = detect_type_facet(front, taxonomy, profile)
    if detected_type:
        detected_facets.setdefault("type", detected_type)

    scope = CardScope()
    tags_resolved: list[str] = []
    tags_unresolved: list[str] = []
    valid_tag_values = taxonomy.all_valid_values()
    resolved_set: set[str] = set()

    for tag in suggested_tags:
        parsed = _parse_tag(tag)
        if parsed is None:
            tags_unresolved.append(f"_unresolved::{tag}")
            continue

        category, value = parsed
        full_tag = f"{category}::{value}"
        if full_tag in valid_tag_values:
            if full_tag not in resolved_set:
                tags_resolved.append(full_tag)
                resolved_set.add(full_tag)
            if scope.get(category) is None:
                scope.set(category, value)
        else:
            tags_unresolved.append(f"_unresolved::{tag}")

    for category, value in detected_facets.items():
        if scope.get(category) is None:
            scope.set(category, value)

    for category, value in scope.summary().items():
        full_tag = f"{category}::{value}"
        if full_tag in valid_tag_values and full_tag not in resolved_set:
            tags_resolved.append(full_tag)
            resolved_set.add(full_tag)

    return scope, tags_resolved, tags_unresolved


def classify_fields(
    *,
    front: str,
    back: str,
    source: str,
    suggested_tags: list[str],
    note_type: str,
    taxonomy: Taxonomy,
    fingerprint: str,
    profile_name: str | None = None,
    profile: ProfileConfig | None = None,
    deck: str | None = None,
    material_origen: str | None = None,
) -> ClassifiedCard:
    """Clasifica campos raw para soportar tarjetas candidatas y notas ya existentes."""
    combined_text = f"{front} {back}"
    scope, tags_resolved, tags_unresolved = build_scope(
        front=front,
        combined_text=combined_text,
        suggested_tags=suggested_tags,
        taxonomy=taxonomy,
        profile=profile,
        deck=deck,
    )

    return ClassifiedCard(
        front=front,
        back=back,
        front_normalized=normalize_text(front),
        back_normalized=normalize_text(back),
        scope=scope,
        tags_resolved=tags_resolved,
        tags_unresolved=tags_unresolved,
        note_type=note_type,
        fingerprint=fingerprint,
        source=source,
        profile=profile_name,
        deck=deck,
        material_origen=material_origen,
    )


def classify(
    card: CandidateCard,
    taxonomy: Taxonomy,
    fingerprint: str,
    *,
    profile_name: str | None = None,
    profile: ProfileConfig | None = None,
) -> ClassifiedCard:
    """Clasifica una tarjeta: detecta scope y resuelve tags contra la taxonomía."""
    return classify_fields(
        front=card.front,
        back=card.back,
        source=card.source,
        suggested_tags=card.suggested_tags,
        note_type=card.note_type,
        taxonomy=taxonomy,
        fingerprint=fingerprint,
        profile_name=profile_name or card.profile,
        profile=profile,
        deck=card.deck,
        material_origen=card.material_origen,
    )
