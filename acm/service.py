"""E6-1 · Capa de servicio única.

La lógica de negocio (sync, resolver cola, routing, búsqueda de registros) vive
ACÁ. `cli.py` y `mcp_server.py` son adaptadores finos que la invocan y solo
formatean la salida (texto/JSON). Antes esta orquestación estaba COPIADA entre
las dos superficies (y las copias ya divergían).
"""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path

from acm.anki.client import AnkiConnectClient, AnkiConnectError
from acm.anki.exporter import export_rows_tsv
from acm.config import Settings
from acm.models import CardScope
from acm.pipeline.similarity import _scope_from_row as scope_from_row  # canónico, único
from acm.store.registry import Registry

__all__ = [
    "scope_from_row",
    "try_anki_client",
    "find_record_by_id_or_prefix",
    "resolve_record",
    "sync_pending",
    "resolve_deck_for_row",
    "backup_registry",
    "undo_batch",
]


def backup_registry(settings: Settings) -> Path:
    """E9-3: copia el registro a ACM_HOME/backups/ antes de una operación masiva."""
    src = settings.db_path_resolved
    backups = src.parent / "backups"
    backups.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dst = backups / f"{src.stem}-{stamp}.db"
    shutil.copy2(src, dst)
    return dst


def try_anki_client(settings: Settings) -> AnkiConnectClient | None:
    """Devuelve un cliente AnkiConnect disponible, o None (degrada con gracia)."""
    try:
        client = AnkiConnectClient(settings.anki.connect_url)
        if client.is_available():
            return client
    except Exception:
        pass
    return None


def find_record_by_id_or_prefix(registry: Registry, record_id: str, candidates):
    """Encuentra un registro por id completo o por prefijo único en `candidates`.

    Devuelve (row, None) si hay match único, o (None, error_dict) si no/ambiguo.
    """
    row = registry.get_by_id(record_id)
    if row:
        return row, None
    matches = [r for r in candidates if r["id"].startswith(record_id)]
    if len(matches) == 1:
        return matches[0], None
    if len(matches) > 1:
        return None, {"error": "Prefijo ambiguo", "matches": [r["id"][:8] for r in matches]}
    return None, {"error": f"No se encontró registro: {record_id}"}


def resolve_record(registry: Registry, record_id: str, action: str) -> dict:
    """E5-1: resuelve un item de la cola con una acción (approve|reject)."""
    normalized = action.strip().lower()
    if normalized not in {"approve", "reject"}:
        return {"error": "action debe ser 'approve' o 'reject'"}

    row, error = find_record_by_id_or_prefix(
        registry, record_id, registry.list_pending_review()
    )
    if error:
        return error

    if normalized == "approve":
        registry.update_action(row["id"], "insert")
        return {"status": "approved", "id": row["id"], "estado": "aprobada",
                "front": row["front_original"]}
    registry.update_action(row["id"], "reject")
    return {"status": "rejected", "id": row["id"], "estado": "descartada"}


def resolve_deck_for_row(row, settings: Settings, client: AnkiConnectClient) -> str:
    """Resuelve el mazo destino de una fila pendiente (target_deck o por scope)."""
    target_deck = row["target_deck"] if "target_deck" in row.keys() else None
    if target_deck:
        return target_deck
    profile_name = row["profile_name"] if "profile_name" in row.keys() else None
    _, profile = settings.get_profile(profile_name)
    return client.resolve_deck(
        scope=scope_from_row(row),
        root_deck=profile.root_deck or settings.anki.default_deck,
        routing_categories=profile.routing_categories,
    )


def sync_pending(
    registry: Registry,
    settings: Settings,
    *,
    anki_client: AnkiConnectClient | None = None,
    export_tsv_path: Path | None = None,
    dry_run: bool = False,
    backup: bool = True,
) -> dict:
    """E4-1/2/3/5 + E9-1/2/3: sube las aprobadas pendientes a Anki, ruteadas e idempotente.

    Solo toma cards 'aprobada' sin anki_note_id; al subir pasan a 'subida' (no
    duplica al re-correr). Anki cerrado → quedan encoladas (+ TSV opcional, E4-5).
    `dry_run=True` previsualiza el plan sin tocar Anki (E9-1). Antes de subir hace
    un backup del registro (E9-3) y etiqueta el lote con un `batch_id` deshacible
    (E9-2).
    """
    pending = registry.list_pending_sync()
    if not pending:
        return {"status": "ok", "message": "Sin tarjetas pendientes", "synced": [],
                "synced_count": 0, "errors": [], "error_count": 0, "anki_available": True}

    owns_client = anki_client is None
    client = anki_client or try_anki_client(settings)
    if client is None:
        result = {"error": "Anki no disponible — las aprobadas quedan encoladas "
                           "(suben al reconectar).", "anki_available": False,
                  "queued": len(pending)}
        if export_tsv_path is not None:
            exported = export_rows_tsv(pending, export_tsv_path)
            result["exported_tsv"] = str(export_tsv_path)
            result["exported_count"] = exported
        return result

    # E9-1: dry-run — previsualizar el plan sin insertar nada.
    if dry_run:
        plan = [
            {"id": row["id"][:8], "deck": resolve_deck_for_row(row, settings, client),
             "front": row["front_original"][:80]}
            for row in pending
        ]
        if owns_client:
            client.close()
        return {"dry_run": True, "would_sync": plan, "count": len(plan), "anki_available": True}

    # E9-3: backup del registro antes de la operación masiva.
    backup_path = backup_registry(settings) if backup else None
    # E9-2: lote deshacible.
    batch_id = datetime.now(timezone.utc).isoformat()

    fm = settings.anki.field_mapping
    synced: list[dict] = []
    errors: list[dict] = []
    for row in pending:
        try:
            tags = row["tags_resolved"].split() if row["tags_resolved"] else []
            deck = resolve_deck_for_row(row, settings, client)
            note_id = client.add_note(
                deck=deck,
                model=row["note_type"] or settings.anki.default_model,
                fields={fm.front: row["front_original"], fm.back: row["back_original"]},
                tags=tags,
            )
            registry.mark_uploaded(row["id"], note_id, batch_id)
            synced.append({"id": row["id"][:8], "deck": deck, "note_id": note_id})
        except AnkiConnectError as e:
            errors.append({"id": row["id"][:8], "error": str(e)})

    if owns_client:
        client.close()

    return {"synced": synced, "synced_count": len(synced), "errors": errors,
            "error_count": len(errors), "anki_available": True, "batch_id": batch_id,
            "backup": str(backup_path) if backup_path else None}


def undo_batch(
    registry: Registry,
    settings: Settings,
    batch_id: str,
    *,
    anki_client: AnkiConnectClient | None = None,
) -> dict:
    """E9-2: deshace un lote de sync — borra las notas de Anki y revierte estado."""
    rows = registry.get_batch(batch_id)
    if not rows:
        return {"error": f"No existe el lote: {batch_id}"}

    note_ids = [row["anki_note_id"] for row in rows if row["anki_note_id"]]
    owns_client = anki_client is None
    client = anki_client or try_anki_client(settings)
    deleted = 0
    if client is not None and note_ids:
        try:
            client.delete_notes(note_ids)
            deleted = len(note_ids)
        except AnkiConnectError as e:
            if owns_client:
                client.close()
            return {"error": f"No se pudieron borrar las notas: {e}", "anki_available": True}
    if owns_client and client is not None:
        client.close()

    reverted = registry.revert_batch(batch_id)
    return {"status": "reverted", "batch_id": batch_id, "deleted_notes": deleted,
            "reverted_records": reverted, "anki_available": client is not None}
