from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from acm.anki.client import AnkiConnectClient
from acm.config import ProfileConfig, Settings, Taxonomy
from acm.pipeline.classifier import classify_fields
from acm.pipeline.normalizer import normalize_text
from acm.pipeline.similarity import (
    DuplicateRecord,
    build_record_from_classified,
    make_fingerprint,
    record_to_index_entry,
)


def _extract_field_value(fields: dict[str, Any], preferred_name: str) -> str:
    if preferred_name in fields:
        return fields[preferred_name].get("value", "")
    if fields:
        first_name = next(iter(fields))
        return fields[first_name].get("value", "")
    return ""


def _extract_back_value(fields: dict[str, Any], preferred_name: str, front_name: str) -> str:
    if preferred_name in fields:
        return fields[preferred_name].get("value", "")

    for field_name, payload in fields.items():
        if field_name == front_name:
            continue
        return payload.get("value", "")
    return ""


def fetch_deck_records(
    client: AnkiConnectClient,
    *,
    deck: str,
    include_subdecks: bool,
    taxonomy: Taxonomy,
    settings: Settings,
    profile_name: str | None = None,
    profile: ProfileConfig | None = None,
) -> list[DuplicateRecord]:
    records: list[DuplicateRecord] = []
    indexed_at = datetime.now(timezone.utc).isoformat()

    for deck_name in client.expand_decks(deck, include_subdecks=include_subdecks):
        note_ids = client.find_notes(f'deck:"{deck_name}"')
        if not note_ids:
            continue

        for note in client.get_notes_info(note_ids):
            fields = note.get("fields", {})
            front = _extract_field_value(fields, settings.anki.field_mapping.front)
            if not front.strip():
                continue

            back = _extract_back_value(
                fields,
                settings.anki.field_mapping.back,
                settings.anki.field_mapping.front,
            )
            note_type = note.get("modelName") or settings.anki.default_model or "Basic"
            fingerprint = make_fingerprint(normalize_text(front), normalize_text(back))
            classified = classify_fields(
                front=front,
                back=back,
                source="anki",
                suggested_tags=note.get("tags", []),
                note_type=note_type,
                taxonomy=taxonomy,
                fingerprint=fingerprint,
                profile_name=profile_name,
                profile=profile,
                deck=deck_name,
            )
            note_id = note.get("noteId")
            records.append(
                build_record_from_classified(
                    classified,
                    candidate_id=f"anki:{note_id}",
                    source="anki",
                    origin_source="anki",
                    deck=deck_name,
                    created_at=indexed_at,
                    anki_note_id=note_id,
                )
            )

    return records


def fetch_recent_notes(
    client: AnkiConnectClient,
    *,
    deck: str,
    include_subdecks: bool,
    days: int = 1,
    settings: Settings,
) -> list[dict]:
    """Fetch notes added in the last N days from a deck via AnkiConnect.

    Returns raw note dicts from AnkiConnect (noteId, fields, tags, modelName).
    """
    notes: list[dict] = []
    for deck_name in client.expand_decks(deck, include_subdecks=include_subdecks):
        note_ids = client.find_notes(f'deck:"{deck_name}" added:{days}')
        if not note_ids:
            continue
        for note in client.get_notes_info(note_ids):
            note["_deck"] = deck_name
            notes.append(note)
    return notes


def records_to_index_entries(records: list[DuplicateRecord]) -> list[dict[str, Any]]:
    return [record_to_index_entry(record) for record in records if record.anki_note_id is not None]
