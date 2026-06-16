"""Flags de calidad pedagógica de tarjetas (E3-1 contrato · E8-2 heurísticas).

Genérico y agnóstico de materia (RNF-7): solo mira forma (longitud, atomicidad,
fuga front↔back, reconocimiento vs recall), nunca contenido de dominio. Reporta;
nunca modifica (no-destructivo). Cada flag: {code, severity, hint}.
"""

from __future__ import annotations

import re

from acm.pipeline.normalizer import normalize_semantic_text, normalize_text

_LIST_MARKER_RE = re.compile(r"(?m)^\s*(?:[-*•·]|\d+[.)])\s+")
_SENTENCE_SPLIT_RE = re.compile(r"[.;\n]+")
_YESNO_RE = re.compile(r"^(s[ií]|no|yes|true|false|verdadero|falso)\b", re.IGNORECASE)

# Umbrales (configurables a futuro; razonables para flashcards de recall).
LONG_ANSWER_CHARS = 320
NON_ATOMIC_SENTENCES = 3
LEAK_MIN_TOKEN_LEN = 6


def _significant_tokens(text: str, *, min_len: int) -> set[str]:
    return {tok for tok in normalize_semantic_text(text).split() if len(tok) >= min_len}


def quality_flags(front: str, back: str) -> list[dict]:
    """Devuelve flags de calidad de una tarjeta (lista vacía = sin problemas)."""
    flags: list[dict] = []
    front_norm = normalize_text(front)
    back_norm = normalize_text(back)

    if not back_norm.strip():
        flags.append({"code": "back_vacio", "severity": "high",
                      "hint": "La tarjeta no tiene respuesta."})
        return flags  # sin back, el resto de checks no aplica

    # Respuesta demasiado larga (cuesta recordar de un tirón).
    if len(back_norm) > LONG_ANSWER_CHARS:
        flags.append({"code": "respuesta_larga", "severity": "medium",
                      "hint": f"Respuesta de {len(back_norm)} chars; considerá dividirla."})

    # No atómica: varios hechos en una sola card (listas o muchas oraciones).
    list_items = len(_LIST_MARKER_RE.findall(back))
    sentences = [s for s in _SENTENCE_SPLIT_RE.split(back_norm) if s.strip()]
    if list_items >= 2 or len(sentences) >= NON_ATOMIC_SENTENCES:
        flags.append({"code": "no_atomica", "severity": "medium",
                      "hint": "Parece cubrir varios hechos; 1 tarjeta = 1 hecho."})

    # Reconocimiento vs recall: respuesta sí/no o trivial → no es recall real.
    if _YESNO_RE.match(back_norm) and len(back_norm) <= 12:
        flags.append({"code": "reconocimiento", "severity": "low",
                      "hint": "Respuesta sí/no: favorece reconocimiento, no recall activo."})

    # Pista filtrada: un término significativo de la respuesta ya está en la
    # pregunta (la trivializa). Conservador: tokens largos compartidos.
    shared = _significant_tokens(front, min_len=LEAK_MIN_TOKEN_LEN) & _significant_tokens(
        back, min_len=LEAK_MIN_TOKEN_LEN
    )
    if shared:
        flags.append({"code": "pista_filtrada", "severity": "low",
                      "hint": f"El front y el back comparten término(s): {sorted(shared)[:3]}."})

    return flags


def suggest_cloze(front: str, back: str) -> bool:
    """E8-3: ¿es candidata a cloze deletion? Back corto y factual (no oración).

    Una card básica cuya respuesta es un término/concepto breve suele rendir
    mejor como cloze. Conservador: 1-4 palabras y sin puntuación de oración.
    """
    answer = normalize_text(back).strip()
    if not answer:
        return False
    words = answer.split()
    if not (1 <= len(words) <= 4):
        return False
    if _YESNO_RE.match(answer):
        return False  # sí/no es reconocimiento, no cloze
    # sin puntuación de oración interna (un término, no una frase larga)
    return not re.search(r"[.;:]", answer)


def suggest_better_version(
    new_front: str, new_back: str, old_front: str, old_back: str
) -> dict:
    """E8-1: ante un casi-duplicado, sugiere si reemplazar la vieja por la nueva.

    Conservador: solo sugiere reemplazar si la nueva es claramente mejor (menos
    flags de calidad, o respuesta más completa sin volverse demasiado larga).
    Nunca decide solo — es una sugerencia para que el usuario apruebe.
    """
    new_flags = quality_flags(new_front, new_back)
    old_flags = quality_flags(old_front, old_back)
    new_bad = len(new_flags)
    old_bad = len(old_flags)

    if new_bad < old_bad:
        return {
            "suggestion": "replace_old_with_new",
            "reason": f"la nueva tiene menos problemas de calidad ({new_bad} vs {old_bad})",
        }

    if new_bad == old_bad:
        new_len = len(new_back.strip())
        old_len = len(old_back.strip())
        new_too_long = any(f["code"] == "respuesta_larga" for f in new_flags)
        if old_len and new_len > old_len * 1.3 and not new_too_long:
            return {
                "suggestion": "replace_old_with_new",
                "reason": "la nueva es más completa (respuesta más detallada)",
            }

    return {"suggestion": "keep_existing", "reason": "la versión existente es igual o mejor"}
