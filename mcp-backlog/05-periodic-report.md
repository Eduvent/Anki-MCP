# 05 — Reporte periódico

**Estado:** ✅ funcional (probado 2026-06-24) · **Prioridad:** baja · **Esfuerzo:** S

## Objetivo

Un resumen semanal/mensual del estado de estudio: top-Again, leeches, retención por tag, cards lentas.

## Por qué

Convierte los datos de 01–02 en un hábito de revisión (la rutina semanal/mensual que recomienda el estudio) sin tener que pedirlo a mano.

## Enfoque

- Componer la salida de 01 + 02 en un reporte legible.
- Candidato a **scheduled task** (p. ej. lunes a la mañana) que deje el reporte y, si hay grupos malos, sugiera abrir modo repair (04).

## Criterios de aceptación

- [x] Reporte con: top cards con más Again, leeches, peor retención por tag, cards más lentas.
- [x] Ejecutable on-demand; sugiere el siguiente paso (`acm_repair`).
- [ ] Programable como scheduled task (pendiente).

## Depende de

`01`, `02`. Mejora con `04`.

## Resultado de prueba (2026-06-24)

✅ Funcional. Devuelve `summary`, `top_again`, `slow_cards` (¡hasta 103s — card de Azure Storage naming!), `worst_retention` y `suggested_next_step` → `acm_repair`.

Bug:
- `summary.leeches = 0` mientras `acm_review_stats` reporta 86 — misma inconsistencia del ticket 01.