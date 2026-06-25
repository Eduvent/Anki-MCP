from __future__ import annotations

from typing import Any

import httpx

from acm.models import CardScope
from acm.pipeline.normalizer import normalize_semantic_text


class AnkiConnectError(Exception):
    pass


class AnkiConnectClient:
    """Wrapper síncrono para AnkiConnect. Preparado para migración a async en v0.2."""

    def __init__(self, url: str = "http://localhost:8765") -> None:
        self.url = url
        self._client = httpx.Client(timeout=5.0)

    def _request(self, action: str, **params: Any) -> Any:
        payload = {"action": action, "version": 6, "params": params}
        try:
            resp = self._client.post(self.url, json=payload)
            resp.raise_for_status()
        except httpx.RequestError as e:
            raise AnkiConnectError(f"No se pudo conectar con AnkiConnect: {e}") from e
        data = resp.json()
        if data.get("error"):
            raise AnkiConnectError(f"AnkiConnect error: {data['error']}")
        return data["result"]

    def is_available(self) -> bool:
        try:
            self._request("version")
            return True
        except AnkiConnectError:
            return False

    def get_decks(self) -> list[str]:
        return self._request("deckNames")

    def get_decks_with_ids(self) -> dict[str, int]:
        return self._request("deckNamesAndIds")

    def find_cards(self, query: str) -> list[int]:
        return self._request("findCards", query=query)

    def cards_info(self, card_ids: list[int]) -> list[dict]:
        return self._request("cardsInfo", cards=card_ids)

    def get_reviews_of_cards(self, card_ids: list[int]) -> dict:
        return self._request("getReviewsOfCards", cards=card_ids)

    def get_model_names(self) -> list[str]:
        return self._request("modelNames")

    def get_model_field_names(self, model: str) -> list[str]:
        """Devuelve los campos reales y su orden para un modelo de Anki."""
        return self._request("modelFieldNames", modelName=model)

    def find_notes(self, query: str) -> list[int]:
        return self._request("findNotes", query=query)

    def get_notes_info(self, note_ids: list[int]) -> list[dict]:
        return self._request("notesInfo", notes=note_ids)

    def expand_decks(self, deck: str, include_subdecks: bool = True) -> list[str]:
        decks = self.get_decks()
        if include_subdecks:
            matches = [name for name in decks if name == deck or name.startswith(deck + "::")]
        else:
            matches = [name for name in decks if name == deck]
        return matches or [deck]

    def add_note(
        self,
        deck: str,
        model: str,
        fields: dict[str, str],
        tags: list[str],
    ) -> int:
        note = {
            "deckName": deck,
            "modelName": model,
            "fields": fields,
            "tags": tags,
            "options": {"allowDuplicate": False},
        }
        return self._request("addNote", note=note)

    def resolve_deck(
        self,
        vendor: str | None = None,
        cert: str | None = None,
        root_deck: str = "Cloud Certs",
        *,
        scope: CardScope | None = None,
        routing_categories: list[str] | None = None,
    ) -> str:
        """Encuentra el mejor deck para una tarjeta según su scope.

        Busca en los decks existentes por coincidencia en el nombre:
        - Primero intenta matchear por las facetas más específicas del perfil
        - Fallback al root_deck
        """
        decks = self.get_decks()
        scope = scope or CardScope(vendor=vendor, cert=cert)
        routing_categories = routing_categories or ["cert", "vendor"]

        # Filtrar solo decks bajo el root
        candidate_decks = [d for d in decks if d == root_deck or d.startswith(root_deck + "::")]

        for category in routing_categories:
            value = scope.get(category)
            if not value:
                continue
            value_normalized = normalize_semantic_text(value)

            for deck in candidate_decks:
                parts = [normalize_semantic_text(part) for part in deck.split("::")]
                if value_normalized in parts[1:]:
                    return deck

            for deck in candidate_decks:
                if value_normalized in normalize_semantic_text(deck):
                    return deck

        return root_deck

    def delete_notes(self, note_ids: list[int]) -> None:
        """Borra notas de Anki por id (E9-2: deshacer un lote subido)."""
        self._request("deleteNotes", notes=note_ids)

    def add_tags(self, note_ids: list[int], tags: str) -> None:
        """Agrega tags a notas existentes. tags es un string separado por espacios."""
        self._request("addTags", notes=note_ids, tags=tags)

    def deck_card_count(self, deck: str, include_subdecks: bool = True) -> int:
        decks = self.expand_decks(deck, include_subdecks=include_subdecks)
        total = 0
        for deck_name in decks:
            total += len(self.find_cards(f'"deck:{deck_name}"'))
        return total

    def remove_tags(self, note_ids: list[int], tags: str) -> None:
        """Remueve tags de notas existentes."""
        self._request("removeTags", notes=note_ids, tags=tags)

    def update_note_fields(self, note_id: int, fields: dict[str, str]) -> None:
        """Actualiza campos de una nota existente."""
        self._request("updateNoteFields", note={"id": note_id, "fields": fields})

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> AnkiConnectClient:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
