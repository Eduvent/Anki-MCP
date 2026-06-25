# Backlog — anki-card-manager (MCP)

Cambios del MCP para cerrar la capa que faltaba: **rendimiento de repaso y reparación**. Cada ticket es un `.md` en esta carpeta.

## Qué hace el MCP hoy

Creación → dedup → clasificación → tag → ruteo → subida + auditoría de higiene:
`acm_annotate`, `acm_auto_classify`, `acm_apply_tags`, `acm_ingest`, `acm_resolve`, `acm_review`, `acm_sync`, `acm_undo`, `acm_audit`, `acm_reorganize`, `acm_find_similar_card`, `acm_taxonomy`, `acm_stats`.

Capa de repaso (nueva, probada 2026-06-24): `acm_decks`, `acm_review_stats`, `acm_retention`, `acm_leech_clusters`, `acm_repair`, `acm_periodic_report`.

## Principio de costo (modelo local vs agente)

| Capa | Herramienta | Costo |
|---|---|---|
| Extracción de datos (leeches, lapses, tiempo, again) | AnkiConnect — query determinista, sin modelo | ~0 |
| Agrupar fallos por similitud/tema | Motor de embeddings que el MCP **ya usa** para dedup | bajo |
| Redactar la reparación de una card | El agente (LLM), solo sobre casos ya agrupados | acotado |

## Tickets

- [x] [01 — Extracción de stats de repaso](01-review-stats-extraction.md) — ✅ funcional (bug menor)
- [x] [02 — Retención por dimensión/tag](02-retention-by-tag.md) — ✅ funcional
- [x] [03 — Clustering de leeches y cards lentas](03-leech-clustering.md) — ✅ corregido
- [x] [04 — Modo repair](04-repair-mode.md) — ✅ corregido
- [x] [05 — Reporte periódico](05-periodic-report.md) — ✅ funcional
- [x] [06 — Enablers (acm_decks + campo Source)](06-enablers.md) — ✅ completo
- [x] [07 — Higiene del front y economía de tokens](07-front-hygiene-token-economy.md) — ✅ corregido

## Resultados de prueba (2026-06-24)

Probadas las 6 tools en vivo (colección real **~1.737 cards**, no 65 — eso era solo el registro del MCP).

**Funcionan:** `acm_decks`, `acm_review_stats`, `acm_retention`, `acm_periodic_report`. La retención por tag ya da el insight clave: peor en `aws::networking::scenario` (0.50), `hashicorp::secrets::command` (0.545), `type::acronym` (0.647), `identity` (0.61).

**Corregido 2026-06-25:** `front`/`back` limpios y compactos, `limit` respetado, miembros por cluster truncados, leeches consistentes entre stats/reporte, `intent` poblado desde texto limpio o fallback `type::*`.