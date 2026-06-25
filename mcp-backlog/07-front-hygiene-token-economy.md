# 07 — Higiene del `front` y economía de tokens

**Estado:** ✅ corregido (probado 2026-06-25) · **Prioridad:** alta (desbloquea 03 y 04) · **Esfuerzo:** S/M · **Origen:** prueba 2026-06-24

## Problema

`acm_leech_clusters` y `acm_repair` superan el límite de tokens (~196k chars) y quedan inusables. Dos causas:

1. **`front` sin limpiar:** cada card lleva el bloque `<style>.card{ font-family… }</style>` de Anki + HTML. Se repite por representante y por cada miembro de cada cluster. Además ensucia los embeddings (ven CSS, no el contenido).
2. **`limit` ignorado:** en `acm_leech_clusters`, `limit=1` y `limit=10` devuelven el mismo tamaño.

## Objetivo

- Stripear HTML/CSS del `front` y devolverlo como `front_excerpt` (igual que `back_excerpt`).
- Respetar `limit` de verdad en todas las tools de repaso.
- Usar el texto limpio también como entrada de los embeddings (mejora clustering y dedup).

## Criterios de aceptación

- [x] `acm_leech_clusters` y `acm_repair` con `limit=10` devuelven una salida compacta.
- [x] `front` de representante y miembros sin `<style>` ni tags HTML (`front_excerpt`).
- [x] `limit` cambia el tamaño de la salida de forma proporcional (`clusters_returned`, `max_members_per_cluster`).

## Depende de

Nada. Es prerequisito de `03` y `04`.