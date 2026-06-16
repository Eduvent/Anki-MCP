from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator


def acm_home() -> Path:
    """Directorio base estable para los datos locales de acm.

    Resuelve a una ruta ABSOLUTA independiente del cwd leyendo la env var
    ``ACM_HOME`` (default ``~/.acm``). Lanzado desde la app de Claude el cwd es
    incierto; anclar el registro/índices/colas/backups a ``ACM_HOME`` evita
    registros partidos y dedup rota (ver E0-1 en BACKLOG.md).
    """
    raw = os.environ.get("ACM_HOME") or "~/.acm"
    home = Path(raw).expanduser()
    home.mkdir(parents=True, exist_ok=True)
    return home.resolve()


# E2-1: NADA de lógica de dominio en el código. Las reglas específicas de materia
# (keywords de vendor/topic, deck→facetas) viven en config/settings.yaml por
# perfil — data-driven, para servir a cualquier materia. Lo único que queda en
# código son patrones de TIPO de pregunta, que son lingüísticos (ES/EN) y
# agnósticos de dominio: "qué es" → definición, "diferencia entre" → comparación.
DEFAULT_PROFILE_TYPE_PATTERNS: dict[str, list[str]] = {
    "acronym": [
        r"\bque significa\b",
        r"\bwhat does .+ stand for\b",
        r"\bsiglas?\b",
    ],
    "definition": [
        r"\bque es\b",
        r"\bwhat is\b",
        r"\bdefine\b",
        r"\bdescribe\b",
        r"\bexplain\b",
        r"\bexplica\b",
    ],
    "comparison": [
        r"\bdiferencia entre\b",
        r"\bdifference between\b",
        r"\bcompare\b",
        r"\bcompara\b",
        r"\bvs\b",
        r"\bversus\b",
    ],
    "command": [
        r"\bcomando\b",
        r"\bcommand\b",
        r"\bcli\b",
        r"\bsyntax\b",
        r"\bvault [a-z0-9_-]+\b",
        r"\bterraform [a-z0-9_-]+\b",
    ],
    "scenario": [
        r"\ba company\b",
        r"\bscenario\b",
        r"\bbest for\b",
        r"\buse case\b",
        r"\bnecesita\b",
        r"\bneeds to\b",
    ],
}


class FieldMapping(BaseModel):
    front: str = "Front"
    back: str = "Back"


class ProfileConfig(BaseModel):
    taxonomy_path: str | None = None
    root_deck: str | None = None
    required_categories: list[str] = Field(default_factory=list)
    routing_categories: list[str] = Field(default_factory=list)
    keyword_rules: dict[str, dict[str, list[str]]] = Field(default_factory=dict)
    deck_tag_rules: dict[str, dict[str, str]] = Field(default_factory=dict)
    topic_keywords: dict[str, list[str]] = Field(default_factory=dict)
    type_patterns: dict[str, list[str]] = Field(default_factory=dict)

    def taxonomy_path_resolved(self, fallback: Path) -> Path:
        return Path(self.taxonomy_path).expanduser() if self.taxonomy_path else fallback


class AnkiConfig(BaseModel):
    connect_url: str = "http://localhost:8765"
    default_deck: str = "Cloud Certs"
    default_model: str = "Básico"
    field_mapping: FieldMapping = FieldMapping()


class AcmConfig(BaseModel):
    # Rutas relativas se resuelven bajo ACM_HOME (no contra el cwd); las
    # absolutas se respetan tal cual. Vacío → default bajo ACM_HOME / packaged.
    db_path: str = "registry.db"
    taxonomy_path: str = ""
    auto_insert: bool = False
    duplicate_threshold: float | None = None
    cluster_threshold: float = 0.90
    similar_lookup_threshold: float = 0.75
    audit_include_subdecks: bool = True
    default_profile: str = "default"
    ollama_url: str = "http://localhost:11434"
    # E1-4 + spike E1-0: el modelo se eligió con datos (ver SPIKE_EMBEDDINGS.md).
    # qwen3-embedding:0.6b es liviano (1024 dim, ~29 ms/texto) y multilingüe:
    # separa paráfrasis ES/EN (gap 0.35), casi como el 8b (gap 0.37) pero 4x más
    # rápido. nomic-embed-text NO separa cross-lingual (gap ~0). Configurable;
    # cambiar de modelo invalida el embedding_cache (keyed por model).
    ollama_model: str = "qwen3-embedding:0.6b"
    use_embeddings: bool = True
    # E2-4: tercer escalón de la escalera de clasificación (determinista → kNN →
    # LLM local → Claude). Off por defecto (requiere un modelo chat en Ollama).
    use_llm_fallback: bool = False
    ollama_chat_model: str = "qwen2.5:0.5b-instruct"

    @model_validator(mode="before")
    @classmethod
    def _compat_duplicate_threshold(cls, data: Any) -> Any:
        if isinstance(data, dict):
            duplicate_threshold = data.get("duplicate_threshold")
            if duplicate_threshold is not None and "cluster_threshold" not in data:
                data = dict(data)
                data["cluster_threshold"] = duplicate_threshold
        return data


class Settings(BaseModel):
    anki: AnkiConfig = AnkiConfig()
    acm: AcmConfig = AcmConfig()
    profiles: dict[str, ProfileConfig] = Field(default_factory=dict)

    @property
    def db_path_resolved(self) -> Path:
        value = (self.acm.db_path or "").strip()
        if not value:
            return acm_home() / "registry.db"
        path = Path(value).expanduser()
        return path if path.is_absolute() else acm_home() / path

    @property
    def taxonomy_path_resolved(self) -> Path:
        value = (self.acm.taxonomy_path or "").strip()
        if not value:
            return _DEFAULT_TAXONOMY_PATH
        path = Path(value).expanduser()
        return path if path.is_absolute() else acm_home() / path

    @property
    def default_profile_name(self) -> str:
        if self.acm.default_profile in self.available_profiles():
            return self.acm.default_profile
        return next(iter(self.available_profiles().keys()))

    def available_profiles(self) -> dict[str, ProfileConfig]:
        if self.profiles:
            merged: dict[str, ProfileConfig] = {}
            for name, profile in self.profiles.items():
                merged[name] = self._merge_with_defaults(profile)
            if self.acm.default_profile not in merged:
                merged[self.acm.default_profile] = self._default_profile()
            return merged
        return {self.acm.default_profile: self._default_profile()}

    def get_profile(self, name: str | None = None) -> tuple[str, ProfileConfig]:
        profiles = self.available_profiles()
        profile_name = name or self.default_profile_name
        if profile_name not in profiles:
            available = ", ".join(sorted(profiles))
            raise ValueError(f"Perfil desconocido: {profile_name}. Disponibles: {available}")
        return profile_name, profiles[profile_name]

    def _default_profile(self) -> ProfileConfig:
        return ProfileConfig(
            taxonomy_path=str(self.taxonomy_path_resolved),
            root_deck=self.anki.default_deck,
            required_categories=["vendor", "topic", "type"],
            routing_categories=["cert", "vendor"],
            # E2-1: sin reglas de dominio en código. Las keywords/deck-rules
            # específicas de materia se cargan desde settings.yaml por perfil.
            # Solo type_patterns (lingüístico, agnóstico de dominio) es default.
            keyword_rules={},
            deck_tag_rules={},
            topic_keywords={},
            type_patterns=DEFAULT_PROFILE_TYPE_PATTERNS,
        )

    def _merge_with_defaults(self, profile: ProfileConfig) -> ProfileConfig:
        base = self._default_profile()
        data = base.model_dump()
        override = profile.model_dump(exclude_none=True)
        for mapping_field in ("keyword_rules", "deck_tag_rules", "topic_keywords", "type_patterns"):
            if getattr(profile, mapping_field):
                data[mapping_field] = {**data[mapping_field], **override[mapping_field]}
        data.update({
            k: v
            for k, v in override.items()
            if k not in {"keyword_rules", "deck_tag_rules", "topic_keywords", "type_patterns"}
            and not (isinstance(v, (list, dict)) and not v)
        })
        return ProfileConfig(**data)


class Taxonomy(BaseModel):
    categories: dict[str, list[str]] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _normalize_input(cls, data: Any) -> Any:
        if data is None:
            return {"categories": {}}
        if isinstance(data, Taxonomy):
            return {"categories": data.categories}
        if isinstance(data, dict):
            if "categories" in data and isinstance(data["categories"], dict):
                raw_categories = data["categories"]
            else:
                raw_categories = data

            categories: dict[str, list[str]] = {}
            for category, values in raw_categories.items():
                if values is None:
                    continue
                if not isinstance(values, list):
                    raise ValueError(f"Taxonomy '{category}' debe ser una lista")
                cleaned = []
                seen: set[str] = set()
                for value in values:
                    value_str = str(value).strip()
                    if not value_str or value_str in seen:
                        continue
                    cleaned.append(value_str)
                    seen.add(value_str)
                categories[str(category).strip()] = cleaned
            return {"categories": categories}
        raise ValueError("Formato de taxonomía inválido")

    def category_names(self) -> list[str]:
        return list(self.categories.keys())

    def values_for(self, category: str) -> list[str]:
        return list(self.categories.get(category, []))

    def append_value(self, category: str, value: str) -> None:
        current = self.categories.setdefault(category, [])
        if value not in current:
            current.append(value)

    def is_valid_tag(self, category: str, value: str) -> bool:
        return value in self.categories.get(category, [])

    def all_valid_values(self) -> set[str]:
        return {
            f"{category}::{value}"
            for category, values in self.categories.items()
            for value in values
        }

    def to_flat_dict(self) -> dict[str, list[str]]:
        return {
            category: list(values)
            for category, values in self.categories.items()
        }

    def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, list[str]]:  # type: ignore[override]
        return self.to_flat_dict()


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f) or {}


_DEFAULT_SETTINGS_PATH = Path(__file__).parent.parent / "config" / "settings.yaml"
_DEFAULT_TAXONOMY_PATH = Path(__file__).parent.parent / "config" / "taxonomy.yaml"


def load_settings(path: Path | None = None) -> Settings:
    p = path or _DEFAULT_SETTINGS_PATH
    if p.exists():
        return Settings(**_load_yaml(p))
    return Settings()


def load_taxonomy(path: Path | None = None) -> Taxonomy:
    p = path or _DEFAULT_TAXONOMY_PATH
    if p.exists():
        return Taxonomy(**_load_yaml(p))
    return Taxonomy()


def load_profile_taxonomy(settings: Settings, profile_name: str | None = None) -> tuple[str, ProfileConfig, Taxonomy]:
    resolved_name, profile = settings.get_profile(profile_name)
    taxonomy = load_taxonomy(profile.taxonomy_path_resolved(settings.taxonomy_path_resolved))
    return resolved_name, profile, taxonomy


def save_taxonomy(taxonomy: Taxonomy, path: Path | None = None) -> None:
    p = path or _DEFAULT_TAXONOMY_PATH
    data = taxonomy.to_flat_dict()
    with open(p, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
