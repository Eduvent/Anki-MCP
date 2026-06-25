from __future__ import annotations

import html
import re
import unicodedata


_HTML_TAG_RE = re.compile(r"<[^>]+>")
_HTML_BLOCK_RE = re.compile(
    r"<(script|style)\b[^>]*>.*?</\1>",
    flags=re.IGNORECASE | re.DOTALL,
)
_MULTI_SPACE_RE = re.compile(r"\s+")
# H1: contenido entre paréntesis/corchetes (aposiciones, alias) — se quita ANTES
# de extraer la clave semántica para que "Azure AD (Entra ID)" y "Azure AD"
# compartan clave. El embedding sigue viendo el texto completo.
_BRACKET_RE = re.compile(r"\([^)]*\)|\[[^\]]*\]")
_NON_WORD_RE = re.compile(r"[^\w\s]")
_LEADING_ARTICLE_RE = re.compile(r"^(?:a|an|the|un|una|unos|unas|el|la|los|las)\s+")
_TRAILING_MEAN_RE = re.compile(r"\s+mean$")
_DEFINITION_PATTERNS = (
    re.compile(r"^(?:what is|what s|define|definition of|tell me the definition of|give me the definition of|explain)\s+(.+)$"),
    re.compile(r"^what does\s+(.+)$"),
    re.compile(r"^(?:que es|cual es la definicion de|dime la definicion de|dame la definicion de|define|explica)\s+(.+)$"),
    re.compile(r"^que significa\s+(.+)$"),
)
_COMPARISON_PATTERNS = (
    re.compile(r"^(?:difference between|compare)\s+(.+?)\s+(?:and|vs|versus)\s+(.+)$"),
    re.compile(r"^(.+?)\s+(?:vs|versus)\s+(.+)$"),
    re.compile(r"^(?:diferencia entre|compara)\s+(.+?)\s+(?:y|vs|versus)\s+(.+)$"),
)
# Acrónimos: "¿qué significan las siglas X?" / "what does X stand for".
# Se chequean ANTES que las de definición ("qué significa X") para no robarles
# el match. Clave estable acronym::X → dups por clave, incluso cross-lingual.
_ACRONYM_PATTERNS = (
    re.compile(r"^que significan?\s+(?:las\s+)?siglas?\s+(?:de\s+)?(.+)$"),
    re.compile(r"^siglas?\s+de\s+(.+)$"),
    re.compile(r"^what (?:does|do)\s+(.+?)\s+stand for\b.*$"),
)


def normalize_text(text: str) -> str:
    """Normaliza texto para comparación y deduplicación.

    Pasos: strip → NFKC unicode → quitar HTML → colapsar espacios → lowercase.
    El original no se modifica; esta función retorna la versión normalizada.
    """
    # 1. Strip whitespace extremo
    text = text.strip()
    # 2. Unicode NFKC (normaliza formas compuestas, full-width chars, etc.)
    text = unicodedata.normalize("NFKC", text)
    # 3. Remover bloques HTML que traen contenido no semántico (CSS/JS) y tags
    text = _HTML_BLOCK_RE.sub(" ", text)
    text = _HTML_TAG_RE.sub(" ", text)
    # 4. Colapsar espacios múltiples (incluye \t, \n, etc.)
    text = _MULTI_SPACE_RE.sub(" ", text).strip()
    # 5. Lowercase para comparación
    text = text.lower()
    return text


def normalize_format(text: str) -> str:
    """E8-4: limpia el formato para ALMACENAR/mostrar (no para comparar).

    A diferencia de ``normalize_text`` (que baja a minúsculas para dedup), esto
    preserva mayúsculas y sentido: decodifica entidades HTML, quita tags, aplica
    NFKC y colapsa espacios. No-destructivo: se ofrece como sugerencia.
    """
    text = strip_html_for_display(text)
    return text


def strip_html_for_display(text: str) -> str:
    """Limpia HTML/CSS/JS preservando mayúsculas para salidas token-eficientes.

    AnkiConnect puede devolver el ``question`` renderizado con bloques
    ``<style>…</style>`` antes del front real. Quitar solo tags deja el CSS como
    texto, ensucia embeddings y explota la salida de MCP. Esta función remueve
    bloques script/style completos, decodifica entidades y colapsa espacios.
    """
    text = html.unescape(text or "")
    text = unicodedata.normalize("NFKC", text)
    text = _HTML_BLOCK_RE.sub(" ", text)
    text = _HTML_TAG_RE.sub(" ", text)
    text = _MULTI_SPACE_RE.sub(" ", text).strip()
    return text


def _strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def normalize_semantic_text(text: str) -> str:
    """Normalización más agresiva para equivalencia semántica heurística."""
    text = _strip_accents(normalize_text(text))
    text = _NON_WORD_RE.sub(" ", text)
    text = _MULTI_SPACE_RE.sub(" ", text).strip()
    return text


def _canonicalize_subject(text: str) -> str:
    subject = normalize_semantic_text(text)
    subject = _TRAILING_MEAN_RE.sub("", subject).strip()
    subject = _LEADING_ARTICLE_RE.sub("", subject)
    subject = _MULTI_SPACE_RE.sub(" ", subject).strip()
    return subject


def detect_intent_and_semantic_key(text: str) -> tuple[str, str | None]:
    """Detecta la intención principal del front y genera una clave canónica si aplica."""
    # H1: quitar aposiciones/alias entre paréntesis antes de derivar la clave.
    normalized = normalize_semantic_text(_BRACKET_RE.sub(" ", text))
    if not normalized:
        return "unknown", None

    for pattern in _ACRONYM_PATTERNS:
        match = pattern.match(normalized)
        if not match:
            continue
        subject = _canonicalize_subject(match.group(1))
        if subject:
            return "acronym", f"acronym::{subject}"

    for pattern in _DEFINITION_PATTERNS:
        match = pattern.match(normalized)
        if not match:
            continue

        subject = _canonicalize_subject(match.group(1))
        if subject:
            return "definition", f"definition::{subject}"

    for pattern in _COMPARISON_PATTERNS:
        match = pattern.match(normalized)
        if not match:
            continue

        left = _canonicalize_subject(match.group(1))
        right = _canonicalize_subject(match.group(2))
        if left and right:
            ordered = sorted((left, right))
            return "comparison", f"comparison::{ordered[0]}::{ordered[1]}"

    return "unknown", None


def detect_intent(text: str) -> str:
    return detect_intent_and_semantic_key(text)[0]


def build_semantic_key(text: str) -> str | None:
    """Extrae una clave semántica estable para preguntas reformuladas.

    La heurística actual se enfoca en preguntas de definición, por ejemplo:
    - "¿Qué es un CDN?"
    - "Dime la definición de un CDN"
    - "What is a CDN?"
    """
    return detect_intent_and_semantic_key(text)[1]
