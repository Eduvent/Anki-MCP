# 01 — Extracción de stats de repaso

**Estado:** ✅ corregido (probado 2026-06-25) · **Prioridad:** alta (base de 02–05) · **Esfuerzo:** M

## Objetivo

Una tool nueva (`acm_review_stats`) que lea de Anki el desempeño real de repaso por carta/nota.

## Por qué

Hoy el MCP no ve nada del estudio. Sin estos datos no hay forma de detectar qué cards fallan, cuáles son leeches ni dónde reparar. `acm_stats` solo cuenta el pipeline de creación.

## Enfoque (capa: AnkiConnect, determinista, sin modelo)

Consultar AnkiConnect:
- Leeches: `findCards` con `tag:leech` y/o `prop:lapses>=N`.
- Por carta: `cardsInfo` / `getReviewsOfCards` → lapses, intervalo, due, suspendida.
- Tiempo de respuesta y again-count desde el historial de reviews.

## Criterios de aceptación

- [x] Devuelve por nota/carta: lapses, again-count, review time medio, estado (leech/suspendida), tags.
- [x] Funciona sin LLM; si Anki está cerrado, error claro.
- [x] Filtrable por mazo, tag, query y `min_lapses`.

## Depende de

Nada. Es la base.

## Resultado de prueba (2026-06-24)

✅ Funcional. 606 cards escaneadas, 510 Again, tiempo medio 22.7s, 5 suspendidas. Filtros OK.

Bugs corregidos (2026-06-25):
- `acm_review_stats` y `acm_periodic_report` usan la misma definición de leech: tag `leech` de Anki OR `lapses >= min_lapses`.
- `front`/`back` se limpian de HTML/CSS/JS y se devuelven como excerpt compacto; `_raw_cards` conserva texto limpio para embeddings sin mutarse por la salida compacta.