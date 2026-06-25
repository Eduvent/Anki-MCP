# 03 — Clustering de leeches y cards lentas

**Estado:** ✅ corregido (probado 2026-06-25) · **Prioridad:** media · **Esfuerzo:** M

## Objetivo

Agrupar las cards problemáticas (leeches + lentas + muchos again) por similitud semántica, para tratar la **causa común** (interferencia entre servicios parecidos) en vez de card por card.

## Por qué

Muchos fallos en cloud vienen de interferencia (NAT GW vs IGW vs VPC Endpoint; NSG vs Azure Firewall). Agrupados, se reparan con tarjetas de contraste dirigidas.

## Enfoque (capa: modelo local — reusar lo existente)

- **Reusar el motor de embeddings que el MCP ya usa para dedup.** No hace falta un modelo nuevo.
- Embeddear las cards problemáticas de 01 y hacer kNN/clustering.
- Etiquetar cada cluster con su eje de tag dominante.

## Criterios de aceptación

- [x] Devuelve clusters con representante + miembros (schema: `clusters[].representative`, `members[]`, `semantic_key`, `intent`, `scope`).
- [x] Reusa embeddings sin dependencia nueva (`embeddings_used` reportado).
- [x] **Barato / cabe en el límite de tokens**: clusters y miembros se limitan y usan excerpts limpios.

## Depende de

`01-review-stats-extraction`.

## Resultado de prueba (2026-06-24)

⚠️ Lógica correcta, pero **inusable por tamaño**: la salida pesa ~196k chars y supera el límite de tokens.

Bugs corregidos (2026-06-25):
- `limit` ahora controla `clusters_returned` y `max_members` limita miembros por cluster.
- Representante y miembros devuelven `front_excerpt`/`back_excerpt` sin `<style>` ni tags HTML.
- `embeddings_used` vuelve en una respuesta compacta.