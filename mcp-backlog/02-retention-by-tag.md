# 02 — Retención por dimensión/tag

**Estado:** ✅ funcional (probado 2026-06-24) · **Prioridad:** alta · **Esfuerzo:** M

## Objetivo

Cruzar el desempeño de repaso (ticket 01) con la taxonomía (`vendor / cert / topic / type`) para saber **dónde** falla Eduardo, no solo qué cards.

## Por qué

El insight más accionable del sistema: *"fallas más en `aws::networking::scenario`"*. Eso dispara sesiones de reparación dirigidas, no por intuición.

## Enfoque (capa: AnkiConnect + agregación local)

- Tomar lapses/again/tiempo de 01 y agrupar por cada eje de tag.
- Calcular retención aproximada por grupo (again = fallo; hard/good/easy = acierto).
- Ranking de los grupos con peor retención y mayor tiempo.

## Criterios de aceptación

- [x] Reporte de retención por `topic`, por `type` y por combinación `vendor::topic::type`.
- [x] Marca grupos por debajo de un umbral configurable (`threshold`).
- [x] Sin LLM (agregación determinista).

## Depende de

`01-review-stats-extraction`.

## Resultado de prueba (2026-06-24)

✅ Funcional y muy útil. 62 grupos, 33 por debajo de 0.80. Peores: `aws::networking::scenario` 0.50, `general::storage::scenario` 0.50, `hashicorp::secrets::command` 0.545, `type::acronym` 0.647, `topic::identity` 0.61. `embeddings_used:false` (correcto). Sin bugs.