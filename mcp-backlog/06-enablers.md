# 06 — Enablers (acm_decks + campo Source)

**Estado:** ✅ corregido (probado 2026-06-25) · **Prioridad:** alta · **Esfuerzo:** S

Dos mejoras chicas e independientes que destraban lo demás.

## 6a — `acm_decks`: listar mazos + conteos

**Objetivo:** una tool que liste los mazos de Anki con su conteo de cards.

**Por qué:** sin esto no había forma de auditar la taxonomía de mazos desde el MCP.

**Enfoque:** AnkiConnect `deckNames` / `deckNamesAndIds` + conteo por mazo. Determinista.

**Aceptación:**
- [x] Devuelve mazos con conteo de cards. ✅

## 6b — Campo `Source` en cada card

**Objetivo:** que toda card lleve fuente (módulo / página / url) en un campo.

**Por qué:** cloud cambia; sin fuente no se puede actualizar una card desactualizada.

**Enfoque:** `source` requerido en `acm_ingest`; mapear a un campo de la nota en el sync.

**Aceptación:**
- [x] `source`/`origin_source` aparecen en las tools de lectura (review_stats/leech_clusters/repair).
- [x] Confirmar que el `sync` escribe el campo en la nota de Anki (test de escritura agregado).

## Resultado de prueba (2026-06-24)

**6a** ✅: 9 mazos, colección real **~1.737 cards** (`Cloud Certs::vendor::cert`: AWS 892 / CLF 446, Azure 72, HashiCorp 170…). Reveló que `acm_stats` (65) cuenta solo el registro del MCP, no la colección.

**6b** ✅: `build_note_fields` escribe `Source`/`Fuente`/`Origen` si el modelo lo tiene, prefiriendo `material_origen` sobre `source`; cubierto por test.