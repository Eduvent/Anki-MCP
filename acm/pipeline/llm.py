"""E2-4 · Fallback de clasificación con LLM local (Ollama).

Tercer escalón de la escalera (RF-C4): determinista → kNN → **LLM local** → Claude.
Solo se invoca para el residuo que los pasos baratos no resolvieron, y solo
propone valores VÁLIDOS de la taxonomía (no inventa). Degrada a {} si Ollama no
responde — nunca rompe el pipeline, solo escala a Claude el caso difícil.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from acm.config import Taxonomy


def ollama_generate(
    prompt: str,
    *,
    ollama_url: str = "http://localhost:11434",
    model: str = "qwen2.5:0.5b-instruct",
    timeout: float = 30.0,
) -> str | None:
    """Llama a /api/generate de Ollama. Devuelve el texto o None si falla."""
    try:
        payload = json.dumps({
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0},
        }).encode()
        req = urllib.request.Request(
            f"{ollama_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
        response = data.get("response")
        return response if isinstance(response, str) else None
    except (urllib.error.URLError, OSError, json.JSONDecodeError, KeyError, ValueError):
        return None


def _extract_json_object(text: str) -> dict | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def classify_with_llm(
    front: str,
    back: str,
    taxonomy: Taxonomy,
    *,
    ollama_url: str = "http://localhost:11434",
    model: str = "qwen2.5:0.5b-instruct",
    categories: list[str] | None = None,
    timeout: float = 30.0,
) -> dict[str, str]:
    """Pide al LLM elegir valores de taxonomía para las categorías dadas.

    Devuelve ``{category: value}`` SOLO con valores válidos en la taxonomía.
    ``{}`` si Ollama no responde o no hay opciones — el residuo escala a Claude.
    """
    target_categories = categories or taxonomy.category_names()
    options = {
        category: taxonomy.values_for(category)
        for category in target_categories
        if taxonomy.values_for(category)
    }
    if not options:
        return {}

    prompt = (
        "Sos un clasificador de flashcards. Elegí para cada categoría UN valor "
        "SOLO de las opciones dadas. Si ninguna opción aplica a una categoría, "
        "omitila. Respondé EXCLUSIVAMENTE un objeto JSON, sin texto extra.\n\n"
        f"FRONT: {front}\n"
        f"BACK: {back}\n\n"
        f"CATEGORÍAS Y OPCIONES VÁLIDAS:\n{json.dumps(options, ensure_ascii=False)}\n\n"
        'Ejemplo de formato: {"vendor": "aws", "topic": "compute"}'
    )
    response = ollama_generate(prompt, ollama_url=ollama_url, model=model, timeout=timeout)
    if not response:
        return {}

    parsed = _extract_json_object(response)
    if parsed is None:
        return {}

    result: dict[str, str] = {}
    for category, value in parsed.items():
        if (
            category in options
            and isinstance(value, str)
            and taxonomy.is_valid_tag(category, value)
        ):
            result[category] = value
    return result
