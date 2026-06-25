# 04 — Modo repair

**Estado:** ✅ corregido (probado 2026-06-25) · **Prioridad:** media · **Esfuerzo:** L

## Objetivo

Dado un leech o un cluster (ticket 03), proponer una reparación concreta en vez de repetir la card igual.

## Por qué

El sistema debe mejorar con el uso. Una card que falla mucho casi siempre es problema de diseño (exceso de info, ambigüedad, falta de prerequisito, interferencia), no de disciplina.

## Enfoque (capa: el agente — única etapa que lo usa)

Solo aquí entra el LLM, sobre casos ya filtrados y agrupados por 01–03. Acciones posibles: dividir, aclarar el prompt, agregar prerequisito, crear tarjeta de contraste, suspender.

La reparación debe respetar el efecto de generación: proponer, que Eduardo confirme/rearticule, recién entonces aplicar vía el pipeline normal (`acm_resolve correct` / `acm_ingest` → `acm_sync`).

## Criterios de aceptación

- [x] Por card/cluster, propone causa probable + acción + card relacionada.
- [x] No aplica nada sin aprobación (no-destructivo).
- [x] Workflow respeta el efecto de generación.

## Depende de

`01`, `02`, `03`.

## Resultado de prueba (2026-06-24)

✅ Lógica correcta y **respeta el efecto de generación**. Schema: `suggestions[]` con `probable_cause`, `suggested_action`, `workflow`, `cluster_context`. Ejemplo real:
- probable_cause: "muchas respuestas Again; respuesta lenta"
- suggested_action: "agregar prerequisito o tarjeta de contraste; aclarar la pista o reducir carga de memoria"
- workflow: "Confirmar con Eduardo; aplicar con acm_resolve(correct) o acm_ingest → acm_sync."

Bugs corregidos (2026-06-25):
- `cluster_context` usa clusters compactos con `front_excerpt`/`back_excerpt` y miembros truncados.
- `limit` limita sugerencias; `acm_repair` ya no arrastra fronts completos.
- El texto se limpia antes de embeddings/intención; si la heurística queda `unknown`, se usa `type::*` clasificado como fallback de `intent`.