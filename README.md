# Anki Card Manager (`acm`)

Capa local entre **Claude** y **Anki**. Vos creás tarjetas conversando con Claude;
`acm` las **deduplica (entre todos los mazos, por similitud semántica), clasifica,
rutea y sube a Anki** — sin copiar/pegar ni organizar a mano.

Es un **connector MCP** (lo usa Claude como herramienta) y también un **CLI**.
Todo corre **local**: embeddings, dedup y clasificación con Ollama; tu material
no sale de tu máquina.

---

## Qué hace

El flujo que automatiza es **crear → revisar → subir**:

1. Claude propone tarjetas y llama a `acm_annotate`. Cada tarjeta vuelve anotada
   (sin subir nada): *¿es duplicada y de cuál?*, *mazo sugerido*, *tags*,
   *confianza*, *flags de calidad*. Mostrás una lista ya deduplicada y clasificada.
2. Aprobás las que van → se persisten en una cola con estado.
3. `acm_sync` las inserta en Anki vía AnkiConnect, ruteadas y taggeadas. **Se
   elimina el copy-paste.**

Principios: **no destructivo** (nunca borra/fusiona sin tu OK), **economía de
tokens** (resuelve local y solo escala a Claude el caso difícil), **cualquier
materia** (cero lógica de dominio en el código; la taxonomía vive en config).

### El motor de deduplicación

Usa **embeddings como recuperador kNN** (coseno) sobre toda la colección, con
atajo exacto por huella. Captura paráfrasis sin solape de palabras —incluso
cross-lingual ES↔EN— que un motor léxico se perdería. Si Ollama no está
disponible, **degrada con gracia** a una capa léxica y lo reporta
(`embeddings_used: false`). Ver `SPIKE_EMBEDDINGS.md` para la elección de modelo
y la calibración.

### La escalera de clasificación (barata → cara)

`determinista (taxonomía/keywords en config) → propagación por kNN (vecinos ya
etiquetados) → LLM local (Ollama, opcional) → Claude (solo el residuo)`.
Solo se auto-aplican tags con **alta confianza**; lo ambiguo se marca para vos.

---

## Requisitos

- **Python ≥ 3.11**
- **[Ollama](https://ollama.com)** con el modelo de embeddings:
  ```bash
  ollama pull qwen3-embedding:0.6b
  # opcional, para el fallback de clasificación con LLM local:
  ollama pull qwen2.5:0.5b-instruct   # o uno más capaz, p.ej. qwen2.5:7b-instruct
  ```
- **Anki** con el addon **[AnkiConnect](https://ankiweb.net/shared/info/2055492159)**
  (solo para subir/auditar; el resto funciona offline).

## Instalación

```bash
# con uv (recomendado)
uv venv && uv pip install -e .

# o con pip
python -m venv .venv && .venv/bin/pip install -e .
```

Esto instala los comandos `acm` (CLI) y `acm-mcp` (servidor MCP).

### Datos y configuración (`ACM_HOME`)

El registro, índices, colas y backups viven en **`ACM_HOME`** (env var, default
`~/.acm`). Es una ruta **absoluta independiente del cwd**, así Claude puede
lanzar el MCP desde cualquier directorio sin partir el registro.

La config está en `config/settings.yaml` (modelo de embeddings, umbrales,
perfiles, reglas de dominio data-driven) y la taxonomía en `config/taxonomy.yaml`.

---

## Uso

### Como connector MCP en Claude

Agregá el servidor a tu configuración de Claude (Desktop/Code), por ejemplo:

```json
{
  "mcpServers": {
    "anki-card-manager": { "command": "acm-mcp" }
  }
}
```

Tools expuestas (19):

| Tool | Qué hace |
|---|---|
| `acm_annotate` | Anota candidatas (dup/mazo/tags/calidad) **sin** subir. El punto de entrada. |
| `acm_ingest` | Persiste un lote (clasifica + dedup + encola). |
| `acm_resolve` | Resuelve la cola con una acción: `approve` / `reject` / `correct`. |
| `acm_review` | Lista la cola de revisión (dups + ambiguas) con su estado. |
| `acm_sync` | Sube las aprobadas a Anki (idempotente; `dry_run`, fallback TSV). |
| `acm_undo` | Deshace un lote de sync (borra esas notas + revierte estado). |
| `acm_audit` | Audita un mazo: `mode=duplicates\|recent\|untagged\|suggest_taxonomy\|maintenance`. |
| `acm_reorganize` | Reorganización masiva one-shot (dry-run + backup previo). |
| `acm_find_similar_card` | Busca similares a un front/back dado. |
| `acm_apply_tags` | Aplica tags a notas nuevas o existentes (por `note_id`). |
| `acm_auto_classify` | Clasificación determinista token-eficiente (sin LLM). |
| `acm_taxonomy` | Consulta/edita la taxonomía: `action=show\|add`. |
| `acm_stats` | Estadísticas + métricas de observabilidad + lotes deshacibles. |
| `acm_decks` | Lista mazos de Anki con conteo de cards. |
| `acm_review_stats` | Lee lapses, Again, tiempo medio, leeches y suspendidas desde el historial de repaso. |
| `acm_retention` | Reporta retención aproximada por tag/dimensión. |
| `acm_leech_clusters` | Agrupa leeches/cards lentas por similitud semántica reutilizando embeddings. |
| `acm_repair` | Sugiere reparaciones no-destructivas para cards problemáticas. |
| `acm_periodic_report` | Resumen on-demand semanal/mensual de estudio. |

### Como CLI

```bash
acm ingest cards.json            # procesa un archivo de tarjetas
acm review                       # ve la cola de revisión
acm approve <id>  /  acm reject <id>
acm sync                         # sube las aprobadas (--export-tsv como fallback)
acm audit-duplicates --deck "Cloud Certs"
acm similar --front "¿Qué es un CDN?"
acm taxonomy show
```

---

## Estados de una tarjeta

`propuesta` (anotada, sin persistir) → `en-revisión` (duplicado posible o tags
ambiguos) → `aprobada` (lista para subir) → `subida` (en Anki) · `descartada`.

Las **aprobadas son la cola offline**: si Anki está cerrado, esperan y suben al
reconectar (o se exportan a TSV).

---

## Desarrollo

```bash
# tests (corren sin Ollama: degradan a léxico)
.venv/bin/python -m pytest -q

# validación del motor de embeddings (elección de modelo + calibración)
PYTHONPATH=. .venv/bin/python scripts/spike_knn.py

# validación end-to-end sobre tu colección real
PYTHONPATH=. .venv/bin/python scripts/validate_real.py
```

Arquitectura: la lógica de negocio vive en `acm/pipeline/` (motor de similitud,
clasificación, calidad, propagación, LLM) y `acm/service.py` (sync, cola, undo,
backup). `acm/mcp_server.py` y `acm/cli.py` son **adaptadores finos**.
`acm/store/registry.py` es el SQLite (registro + índice + cache de embeddings).
