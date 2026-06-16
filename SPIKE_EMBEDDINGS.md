# Spike E1-0 — Embeddings como recuperador (kNN)

> Validación empírica de la **Opción A** (embeddings-como-recuperador) antes de
> retirar el motor léxico pesado (E1-7). Decide modelo (E1-4) y umbrales (E1-6).
> Reproducir con: `PYTHONPATH=. .venv/bin/python scripts/spike_knn.py`

## Pregunta

¿El coseno de embeddings separa **paráfrasis** (misma pregunta, otras palabras,
incluso cross-lingual ES↔EN) de pares **no relacionados**, lo suficiente para
usarlo como mecanismo de recuperación de candidatos en lugar del blocking léxico?

## Resultados (4 pares paráfrasis ES/EN + 3 no relacionados)

| Modelo | dim | latencia (warm) | paráfrasis (min–avg) | no-relac. (max–avg) | separación | veredicto |
|---|---|---|---|---|---|---|
| nomic-embed-text | 768 | ~15 ms/texto | 0.40 – 0.46 | 0.45 – 0.40 | **−0.06** | ❌ débil |
| nomic-embed-text +prefix | 768 | ~10 ms/texto | 0.56 – 0.59 | 0.56 – 0.48 | +0.00 | ❌ débil |
| **qwen3-embedding:0.6b** | **1024** | **~29 ms/texto** | **0.55 – 0.58** | **0.20 – 0.12** | **+0.35** | ✅ **elegido** |
| qwen3-embedding:8b | 4096 | ~116 ms/texto | 0.61 – 0.65 | 0.23 – 0.18 | +0.37 | ✅ viable (4x lento) |

- **3 de 4** paráfrasis NO compartían ningún `block_key` léxico → el motor léxico
  jamás las habría comparado. El kNN sí las recupera.
- **nomic-embed-text** (el "modelo liviano" que sugería la revisión) **falla en
  cross-lingual**: paráfrasis ES↔EN y no-relacionados se solapan (gap ~0). Es
  primariamente inglés. Descartado para este uso (el usuario estudia en ES+EN).
- **qwen3-embedding:0.6b** logra casi la misma separación que el 8b (0.35 vs 0.37)
  a **¼ de la latencia y ¼ de la dimensión** → es el default (E1-4).

## Decisión de modelo (E1-4)

`ollama_model: qwen3-embedding:0.6b` (configurable en `settings.yaml`).
Liviano y multilingüe. Cambiar de modelo **invalida el `embedding_cache`**
(está keyed por `model`, así que no se mezclan dimensiones).

## Calibración de umbrales (E1-6)

El coseno crudo NO está en la escala de las señales léxicas (0.80–1.0). Con
qwen3:0.6b: paráfrasis ≈ 0.55–0.64, no-relacionados ≈ 0.07–0.20. Sin calibrar,
`cluster_threshold=0.90` jamás agruparía una paráfrasis a coseno 0.60.

`calibrate_cosine()` (en `similarity.py`) mapea coseno→escala léxica con anclas:

| coseno crudo | score calibrado | significado |
|---|---|---|
| ≤ 0.20 | 0.00 | no relacionado |
| 0.40 | 0.75 | "similar" (`similar_lookup_threshold`) |
| 0.55 | 0.90 | "cluster" (duplicado fuerte) |
| 0.68 | 0.97 | casi idéntico |
| 1.00 | 1.00 | idéntico |

Así un único umbral (`cluster_threshold`/`similar_lookup_threshold`) sirve para
señales léxicas Y de embeddings. `compare_records` hace `max(léxico, calibrado)`
(E0-2): los embeddings suman recall sin pisar la léxica fuerte.

**Si cambiás de modelo, recalibrá** estas anclas corriendo el spike y ajustando
`_COSINE_CALIBRATION`.

## Validación sobre datos reales (429 notas, `scripts/validate_real.py`)

- 1ª corrida (embeber + cachear 429 notas): **11.0 s** (~26 ms/nota).
- 2ª corrida (cache hit por fingerprint): **0.05 s** → **218× speedup** (E1-3 ✓).
- Clustering kNN cross-deck: **3 clusters** reales, todos vía `embedding_match`
  (paráfrasis ES de modelos de nube IaaS/PaaS; preguntas Terraform reformuladas).

## Veredicto

✅ **Opción A confirmada.** El kNN por coseno recupera paráfrasis que el léxico
pierde, con separación amplia usando un modelo liviano multilingüe y latencia
dominada por un cache por fingerprint. Habilita E1-7 (retirar el léxico pesado).

### Limitación conocida (futuro)

El clustering full-collection es O(M²·dim). Para 429 notas ≈ 13 s (aceptable; es
un audit explícito, no el camino de creación que es O(M) y <0.5 s). Para
colecciones grandes (miles), considerar un índice ANN o vectorización numpy.
