"""Ollama embedding client for semantic similarity."""

from __future__ import annotations

import json
import math
import urllib.request
import urllib.error
from typing import Sequence


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def get_embeddings(
    texts: list[str],
    *,
    ollama_url: str = "http://localhost:11434",
    model: str = "qwen3-embedding:0.6b",
    timeout: float = 30.0,
) -> list[list[float]] | None:
    """Compute embeddings for a batch of texts via Ollama API.

    Returns None if Ollama is unavailable or errors out.
    """
    if not texts:
        return []
    try:
        payload = json.dumps({"model": model, "input": texts}).encode()
        req = urllib.request.Request(
            f"{ollama_url}/api/embed",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
        embeddings = data.get("embeddings")
        if embeddings and len(embeddings) == len(texts):
            return embeddings
        return None
    except (urllib.error.URLError, OSError, json.JSONDecodeError, KeyError):
        return None


def get_embedding(
    text: str,
    *,
    ollama_url: str = "http://localhost:11434",
    model: str = "qwen3-embedding:0.6b",
    timeout: float = 15.0,
) -> list[float] | None:
    result = get_embeddings([text], ollama_url=ollama_url, model=model, timeout=timeout)
    if result and len(result) == 1:
        return result[0]
    return None


def embeddings_available(
    *,
    ollama_url: str = "http://localhost:11434",
    model: str = "qwen3-embedding:0.6b",
    timeout: float = 4.0,
) -> bool:
    """Sondea si Ollama responde con el modelo configurado (E0-3 / E0-2).

    Permite reportar ``embeddings_usados`` con honestidad: si Ollama está caído
    o el modelo no está disponible, las tools degradan a léxico y lo informan en
    vez de hacerlo en silencio.
    """
    return get_embeddings(["ping"], ollama_url=ollama_url, model=model, timeout=timeout) is not None
