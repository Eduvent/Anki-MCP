from __future__ import annotations

from typing import Any

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class CandidateCard(BaseModel):
    front: str
    back: str
    source: str  # "chatgpt", "claude", "manual"
    suggested_tags: list[str] = []
    note_type: str = "Basic"
    meta: dict = {}
    profile: str | None = None
    deck: str | None = None
    material_origen: str | None = None  # E3-2: PDF/sección de origen, para rastreo

    @field_validator("front", "back")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("field cannot be empty")
        return v


class CardScope(BaseModel):
    """C-1: `facets` es la ÚNICA fuente de verdad del scope.

    vendor/topic/cert ya no son campos sincronizados (eso causaba bugs de "cuál
    es la fuente"); ahora son propiedades de solo-lectura derivadas de facets.
    Se siguen aceptando como kwargs legacy al construir (se pliegan en facets).
    """

    facets: dict[str, str] = Field(default_factory=dict)

    # Tolera kwargs legacy vendor/topic/cert (el before-validator los pliega).
    model_config = {"extra": "ignore"}

    @model_validator(mode="before")
    @classmethod
    def _coerce_facets(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        facets: dict[str, str] = {}
        raw_facets = data.get("facets") or {}
        if isinstance(raw_facets, dict):
            for key, value in raw_facets.items():
                if value is None:
                    continue
                value_str = str(value).strip()
                if value_str:
                    facets[str(key).strip()] = value_str

        for legacy_key in ("vendor", "topic", "cert"):
            value = data.get(legacy_key)
            if value is None:
                continue
            value_str = str(value).strip()
            if value_str:
                facets.setdefault(legacy_key, value_str)

        return {"facets": facets}

    @property
    def vendor(self) -> str | None:
        return self.facets.get("vendor")

    @property
    def topic(self) -> str | None:
        return self.facets.get("topic")

    @property
    def cert(self) -> str | None:
        return self.facets.get("cert")

    def get(self, category: str) -> str | None:
        return self.facets.get(category)

    def set(self, category: str, value: str | None) -> None:
        if value is None or not str(value).strip():
            self.facets.pop(category, None)
        else:
            self.facets[category] = str(value).strip()

    def summary(self) -> dict[str, str]:
        return dict(self.facets)


class ClassifiedCard(BaseModel):
    # Originals preserved for display
    front: str
    back: str
    # Normalized versions for comparison
    front_normalized: str
    back_normalized: str
    scope: CardScope
    tags_resolved: list[str]        # tags validados contra taxonomía
    tags_unresolved: list[str] = [] # tags que no están en taxonomía
    note_type: str
    fingerprint: str                # sha256 para dedupe exacto
    source: str
    profile: str | None = None
    deck: str | None = None
    material_origen: str | None = None  # E3-2: procedencia (PDF/sección)


class AuditDecision(BaseModel):
    card: ClassifiedCard
    action: Literal["insert", "possible_duplicate", "reject"]
    reason: str
    matches: list[str] = []  # IDs/fingerprints de tarjetas similares encontradas
    # E1-5: detalle de cada match fuerte (score + razón + front/deck) para que la
    # cola de revisión muestre contra qué choca cada card, sin recomputar.
    match_details: list[dict] = []
