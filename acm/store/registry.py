from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator

from acm.models import AuditDecision


_SCHEMA = """
CREATE TABLE IF NOT EXISTS processed_cards (
    id TEXT PRIMARY KEY,
    fingerprint TEXT NOT NULL,
    front_original TEXT NOT NULL,
    back_original TEXT NOT NULL,
    front_normalized TEXT NOT NULL,
    back_normalized TEXT NOT NULL,
    scope_json TEXT NOT NULL DEFAULT '{}',
    note_type TEXT NOT NULL DEFAULT 'Basic',
    tags_resolved TEXT NOT NULL DEFAULT '',
    action_taken TEXT NOT NULL,
    anki_note_id INTEGER,
    created_at TEXT NOT NULL,
    source TEXT NOT NULL,
    profile_name TEXT,
    target_deck TEXT,
    match_json TEXT NOT NULL DEFAULT '[]',
    material_origen TEXT,
    status TEXT NOT NULL DEFAULT 'aprobada',
    reason TEXT,
    sync_batch TEXT,
    audit_action TEXT,
    ingest_batch TEXT
);

CREATE INDEX IF NOT EXISTS idx_fingerprint ON processed_cards(fingerprint);
CREATE INDEX IF NOT EXISTS idx_action ON processed_cards(action_taken);

CREATE TABLE IF NOT EXISTS indexed_notes (
    note_id INTEGER PRIMARY KEY,
    deck_name TEXT NOT NULL,
    note_type TEXT NOT NULL,
    front_original TEXT NOT NULL,
    back_original TEXT NOT NULL,
    front_normalized TEXT NOT NULL,
    back_normalized TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    scope_json TEXT NOT NULL DEFAULT '{}',
    intent TEXT NOT NULL DEFAULT 'unknown',
    semantic_key TEXT,
    anchor_tokens TEXT NOT NULL DEFAULT '',
    trigram_signature TEXT NOT NULL DEFAULT '',
    content_tokens TEXT NOT NULL DEFAULT '',
    back_tokens TEXT NOT NULL DEFAULT '',
    char_trigrams TEXT NOT NULL DEFAULT '',
    block_keys TEXT NOT NULL DEFAULT '',
    embedding_vector TEXT NOT NULL DEFAULT '',
    indexed_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_indexed_notes_deck ON indexed_notes(deck_name);
CREATE INDEX IF NOT EXISTS idx_indexed_notes_semantic_key ON indexed_notes(semantic_key);

CREATE TABLE IF NOT EXISTS indexed_decks (
    deck_name TEXT PRIMARY KEY,
    include_subdecks INTEGER NOT NULL,
    last_refreshed_at TEXT NOT NULL,
    note_count INTEGER NOT NULL DEFAULT 0
);

-- E1-3/E1-4: cache de vectores por (fingerprint, model). Una sola fuente para
-- TODOS los pools (registro, índice, batch). Keyed por model → cambiar de
-- modelo de embeddings invalida el cache automáticamente (no mezcla dims).
CREATE TABLE IF NOT EXISTS embedding_cache (
    fingerprint TEXT NOT NULL,
    model TEXT NOT NULL,
    dim INTEGER NOT NULL,
    vector TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (fingerprint, model)
);
"""


# E5-2: estados del ciclo de vida de una tarjeta, persistidos en `status`.
#   propuesta  → (anotada, sin persistir; estado conceptual de acm_annotate)
#   en-revision → duplicado posible o clasificación ambigua (tags sin resolver)
#   aprobada    → válida y lista para subir
#   subida      → ya insertada en Anki (tiene anki_note_id)
#   descartada  → rechazada
STATUS_IN_REVIEW = "en-revision"
STATUS_APPROVED = "aprobada"
STATUS_UPLOADED = "subida"
STATUS_DISCARDED = "descartada"
STATUS_DELETED_IN_ANKI = "borrada-en-anki"  # H3: la nota se borró a mano en Anki

# Estados "removidos": no cuentan como duplicado vivo para la dedup (H3/H4).
REMOVED_STATUSES = (STATUS_DISCARDED, STATUS_DELETED_IN_ANKI)


def derive_status(action: str, anki_note_id: int | None, has_unresolved: bool) -> str:
    """Mapea (acción de auditoría, anki_note_id, ambigüedad) → estado de ciclo de vida."""
    if action == "reject":
        return STATUS_DISCARDED
    if action == "possible_duplicate":
        return STATUS_IN_REVIEW
    # action == "insert"
    if anki_note_id is not None:
        return STATUS_UPLOADED
    if has_unresolved:
        return STATUS_IN_REVIEW  # E5-1: ambiguo → a la cola, no auto-aprobado
    return STATUS_APPROVED


class Registry:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(_SCHEMA)
            self._migrate(conn)

    def _migrate(self, conn: sqlite3.Connection) -> None:
        pc_columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(processed_cards)").fetchall()
        }
        if "note_type" not in pc_columns:
            conn.execute(
                "ALTER TABLE processed_cards ADD COLUMN note_type TEXT NOT NULL DEFAULT 'Basic'"
            )

        idx_columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(indexed_notes)").fetchall()
        }
        new_idx_cols = {
            "intent": "TEXT NOT NULL DEFAULT 'unknown'",
            "content_tokens": "TEXT NOT NULL DEFAULT ''",
            "back_tokens": "TEXT NOT NULL DEFAULT ''",
            "char_trigrams": "TEXT NOT NULL DEFAULT ''",
            "block_keys": "TEXT NOT NULL DEFAULT ''",
            "scope_json": "TEXT NOT NULL DEFAULT '{}'",
            "embedding_vector": "TEXT NOT NULL DEFAULT ''",
        }
        for col_name, col_def in new_idx_cols.items():
            if col_name not in idx_columns:
                conn.execute(f"ALTER TABLE indexed_notes ADD COLUMN {col_name} {col_def}")

        new_pc_cols = {
            "scope_json": "TEXT NOT NULL DEFAULT '{}'",
            "profile_name": "TEXT",
            "target_deck": "TEXT",
            "match_json": "TEXT NOT NULL DEFAULT '[]'",
            "material_origen": "TEXT",
            "status": "TEXT NOT NULL DEFAULT 'aprobada'",
            "reason": "TEXT",
            "sync_batch": "TEXT",
            "audit_action": "TEXT",
            "ingest_batch": "TEXT",
        }
        status_added = "status" not in pc_columns
        audit_action_added = "audit_action" not in pc_columns
        for col_name, col_def in new_pc_cols.items():
            if col_name not in pc_columns:
                conn.execute(f"ALTER TABLE processed_cards ADD COLUMN {col_name} {col_def}")

        # E5-2: backfill de estados para filas legacy desde action_taken/anki_note_id.
        if status_added:
            conn.execute(
                f"UPDATE processed_cards SET status = CASE "
                f"  WHEN action_taken = 'reject' THEN '{STATUS_DISCARDED}' "
                f"  WHEN action_taken = 'possible_duplicate' THEN '{STATUS_IN_REVIEW}' "
                f"  WHEN anki_note_id IS NOT NULL THEN '{STATUS_UPLOADED}' "
                f"  ELSE '{STATUS_APPROVED}' END"
            )

        # Telemetría (§11): backfill del veredicto original desde action_taken
        # (aproximado para filas legacy; exacto de aquí en más vía insert()).
        if audit_action_added:
            conn.execute(
                "UPDATE processed_cards SET audit_action = action_taken "
                "WHERE audit_action IS NULL"
            )

        # C-1: retirar columnas legacy scope_vendor/topic/cert (facets+scope_json
        # es la única fuente). Backfill scope_json desde las columnas antes de
        # dropearlas, para no perder datos de DBs viejas.
        for table in ("processed_cards", "indexed_notes"):
            existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
            legacy = [c for c in ("scope_vendor", "scope_topic", "scope_cert") if c in existing]
            if not legacy:
                continue
            rows = conn.execute(
                f"SELECT rowid AS _rid, scope_json, {', '.join(legacy)} FROM {table}"
            ).fetchall()
            for row in rows:
                try:
                    facets = json.loads(row["scope_json"]) if row["scope_json"] else {}
                except (json.JSONDecodeError, TypeError):
                    facets = {}
                if not isinstance(facets, dict):
                    facets = {}
                changed = False
                for category in ("vendor", "topic", "cert"):
                    col = f"scope_{category}"
                    if col in legacy and row[col] and category not in facets:
                        facets[category] = row[col]
                        changed = True
                if changed:
                    conn.execute(
                        f"UPDATE {table} SET scope_json = ? WHERE rowid = ?",
                        (json.dumps(facets, sort_keys=True), row["_rid"]),
                    )
            conn.execute("DROP INDEX IF EXISTS idx_scope")
            conn.execute("DROP INDEX IF EXISTS idx_indexed_notes_scope")
            for col in legacy:
                conn.execute(f"ALTER TABLE {table} DROP COLUMN {col}")

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def insert(
        self,
        decision: AuditDecision,
        anki_note_id: int | None = None,
        ingest_batch: str | None = None,
    ) -> str:
        """Registra una tarjeta procesada. Retorna el ID generado.

        `audit_action` guarda el veredicto ORIGINAL (inmutable, para telemetría);
        `ingest_batch` etiqueta el lote de ingesta (para deshacer, §5).
        """
        card = decision.card
        record_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        tags_str = " ".join(card.tags_resolved)
        scope_json = json.dumps(card.scope.summary(), sort_keys=True)
        match_json = json.dumps(decision.match_details, ensure_ascii=False)
        status = derive_status(
            decision.action, anki_note_id, has_unresolved=bool(card.tags_unresolved)
        )
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO processed_cards
                    (id, fingerprint, front_original, back_original,
                     front_normalized, back_normalized,
                     scope_json, note_type,
                     tags_resolved, action_taken, anki_note_id, created_at, source,
                     profile_name, target_deck, match_json, material_origen, status, reason,
                     audit_action, ingest_batch)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record_id,
                    card.fingerprint,
                    card.front,
                    card.back,
                    card.front_normalized,
                    card.back_normalized,
                    scope_json,
                    card.note_type,
                    tags_str,
                    decision.action,
                    anki_note_id,
                    now,
                    card.source,
                    card.profile,
                    card.deck,
                    match_json,
                    card.material_origen,
                    status,
                    decision.reason,
                    decision.action,  # audit_action (inmutable)
                    ingest_batch,
                ),
            )
        return record_id

    def find_by_fingerprint(self, fingerprint: str) -> sqlite3.Row | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM processed_cards WHERE fingerprint = ?",
                (fingerprint,),
            ).fetchone()
        return row

    def list_pending_review(self) -> list[sqlite3.Row]:
        """E5-1: cola de revisión = duplicados posibles + ambiguos (status en-revision)."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM processed_cards WHERE status = ? ORDER BY created_at DESC",
                (STATUS_IN_REVIEW,),
            ).fetchall()
        return rows

    def list_processed_cards(self) -> list[sqlite3.Row]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM processed_cards ORDER BY created_at ASC"
            ).fetchall()
        return rows

    def list_active_cards(self) -> list[sqlite3.Row]:
        """Cards 'vivas' para la dedup: excluye descartada y borrada-en-anki (H3/H4)."""
        placeholders = ",".join("?" for _ in REMOVED_STATUSES)
        with self._conn() as conn:
            return conn.execute(
                f"SELECT * FROM processed_cards WHERE status NOT IN ({placeholders}) "
                "ORDER BY created_at ASC",
                REMOVED_STATUSES,
            ).fetchall()

    def list_uploaded_note_ids(self) -> list[tuple[str, int]]:
        """(record_id, anki_note_id) de las cards en estado 'subida' (para reconciliar)."""
        with self._conn() as conn:
            return [
                (row["id"], row["anki_note_id"])
                for row in conn.execute(
                    "SELECT id, anki_note_id FROM processed_cards "
                    "WHERE status = ? AND anki_note_id IS NOT NULL",
                    (STATUS_UPLOADED,),
                ).fetchall()
            ]

    def mark_deleted_in_anki(self, record_ids: list[str]) -> int:
        """H3: marca registros como 'borrada-en-anki' (su nota ya no existe)."""
        if not record_ids:
            return 0
        with self._conn() as conn:
            conn.executemany(
                "UPDATE processed_cards SET status = ? WHERE id = ?",
                [(STATUS_DELETED_IN_ANKI, rid) for rid in record_ids],
            )
        return len(record_ids)

    def list_pending_sync(self) -> list[sqlite3.Row]:
        """Tarjetas aprobadas listas para subir (status aprobada, sin anki_note_id).

        Las ambiguas (en-revision) NO se sincronizan hasta que el usuario las
        resuelva — E5-1 / decisión #3 (auto solo con alta confianza)."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM processed_cards WHERE status = ? AND anki_note_id IS NULL ORDER BY created_at ASC",
                (STATUS_APPROVED,),
            ).fetchall()
        return rows

    def update_action(self, record_id: str, action: str, anki_note_id: int | None = None) -> None:
        status = derive_status(action, anki_note_id, has_unresolved=False)
        with self._conn() as conn:
            conn.execute(
                "UPDATE processed_cards SET action_taken = ?, anki_note_id = ?, status = ? WHERE id = ?",
                (action, anki_note_id, status, record_id),
            )

    def get_by_id(self, record_id: str) -> sqlite3.Row | None:
        with self._conn() as conn:
            return conn.execute(
                "SELECT * FROM processed_cards WHERE id = ?", (record_id,)
            ).fetchone()

    def find_by_id_prefix(self, prefix: str) -> list[sqlite3.Row]:
        """Busca por prefijo de id en TODOS los estados (no solo la cola) — §6."""
        with self._conn() as conn:
            return conn.execute(
                "SELECT * FROM processed_cards WHERE id LIKE ? ORDER BY created_at DESC",
                (prefix + "%",),
            ).fetchall()

    def update_card_fields(self, record_id: str, card) -> None:
        """Upsert por contenido (§7): actualiza campos mutables (note_type, tags,
        scope, deck, procedencia) sin tocar id/estado/anki_note_id/created_at."""
        scope_json = json.dumps(card.scope.summary(), sort_keys=True)
        with self._conn() as conn:
            conn.execute(
                "UPDATE processed_cards SET note_type=?, tags_resolved=?, scope_json=?, "
                "target_deck=?, material_origen=? WHERE id=?",
                (card.note_type, " ".join(card.tags_resolved), scope_json,
                 card.deck, card.material_origen, record_id),
            )

    def update_from_decision(self, record_id: str, decision: AuditDecision) -> None:
        """E5-3: reescribe una card existente con una decisión fresca (corrección).

        Re-deduplicada y re-clasificada; vuelve a estado de cola/aprobada y limpia
        anki_note_id/sync_batch (no estaba subida o se corrigió antes de subir).
        """
        card = decision.card
        status = derive_status(decision.action, None, has_unresolved=bool(card.tags_unresolved))
        with self._conn() as conn:
            conn.execute(
                """
                UPDATE processed_cards SET
                    fingerprint=?, front_original=?, back_original=?,
                    front_normalized=?, back_normalized=?, scope_json=?, note_type=?,
                    tags_resolved=?, action_taken=?, anki_note_id=NULL,
                    match_json=?, material_origen=?, status=?, reason=?, sync_batch=NULL
                WHERE id=?
                """,
                (
                    card.fingerprint, card.front, card.back,
                    card.front_normalized, card.back_normalized,
                    json.dumps(card.scope.summary(), sort_keys=True), card.note_type,
                    " ".join(card.tags_resolved), decision.action,
                    json.dumps(decision.match_details, ensure_ascii=False),
                    card.material_origen, status, decision.reason, record_id,
                ),
            )

    def mark_uploaded(self, record_id: str, anki_note_id: int, batch_id: str) -> None:
        """E9-2: marca subida y la asocia a un lote de sync (para deshacer)."""
        with self._conn() as conn:
            conn.execute(
                "UPDATE processed_cards SET action_taken='insert', anki_note_id=?, "
                "status=?, sync_batch=? WHERE id=?",
                (anki_note_id, STATUS_UPLOADED, batch_id, record_id),
            )

    def list_sync_batches(self) -> list[sqlite3.Row]:
        with self._conn() as conn:
            return conn.execute(
                "SELECT sync_batch, COUNT(*) AS note_count, MIN(created_at) AS created_at "
                "FROM processed_cards WHERE sync_batch IS NOT NULL "
                "GROUP BY sync_batch ORDER BY sync_batch DESC"
            ).fetchall()

    def get_batch(self, batch_id: str) -> list[sqlite3.Row]:
        with self._conn() as conn:
            return conn.execute(
                "SELECT * FROM processed_cards WHERE sync_batch = ?", (batch_id,)
            ).fetchall()

    def revert_batch(self, batch_id: str) -> int:
        """E9-2: revierte un lote — vuelve a 'aprobada' y limpia note_id/batch."""
        with self._conn() as conn:
            cur = conn.execute(
                "UPDATE processed_cards SET status=?, anki_note_id=NULL, sync_batch=NULL "
                "WHERE sync_batch=?",
                (STATUS_APPROVED, batch_id),
            )
            return cur.rowcount

    def metrics(self) -> dict:
        """E9-4: métricas por estado + auto-resueltas vs escaladas al usuario."""
        with self._conn() as conn:
            by_status = {
                row["status"]: row["c"]
                for row in conn.execute(
                    "SELECT status, COUNT(*) AS c FROM processed_cards GROUP BY status"
                )
            }
        auto_resolved = by_status.get(STATUS_APPROVED, 0) + by_status.get(STATUS_UPLOADED, 0)
        escalated = by_status.get(STATUS_IN_REVIEW, 0)
        return {
            "by_status": by_status,
            "auto_resolved": auto_resolved,
            "escalated_to_user": escalated,
            "discarded": by_status.get(STATUS_DISCARDED, 0),
        }

    def precision_metrics(self) -> dict:
        """§11: telemetría de precisión de dedup (match propuesto → resolución).

        Matriz veredicto_original (audit_action) → estado final, y un proxy de
        falsos positivos: de las flags `possible_duplicate` ya resueltas, cuántas
        terminó el usuario CONSERVANDO (aprobada/subida → la flag sobró = FP) vs
        DESCARTANDO (descartada → dup confirmado = TP). Sirve para auto-calibrar.
        """
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT audit_action, status, COUNT(*) AS c FROM processed_cards "
                "WHERE audit_action IS NOT NULL GROUP BY audit_action, status"
            ).fetchall()
        matrix: dict[str, dict[str, int]] = {}
        for row in rows:
            matrix.setdefault(row["audit_action"], {})[row["status"]] = row["c"]

        flagged = matrix.get("possible_duplicate", {})
        kept = flagged.get(STATUS_APPROVED, 0) + flagged.get(STATUS_UPLOADED, 0)
        discarded = flagged.get(STATUS_DISCARDED, 0)
        pending = flagged.get(STATUS_IN_REVIEW, 0)
        resolved = kept + discarded
        return {
            "transition_matrix": matrix,
            "dup_flag": {
                "kept_overridden": kept,        # usuario conservó pese a la flag → FP
                "confirmed_discarded": discarded,  # usuario descartó → dup real → TP
                "pending": pending,
                "resolved": resolved,
                "false_positive_proxy": round(kept / resolved, 3) if resolved else None,
            },
        }

    def delete_record(self, record_id: str) -> bool:
        """§5: borrado físico de un registro (purga). True si borró algo."""
        with self._conn() as conn:
            cur = conn.execute("DELETE FROM processed_cards WHERE id = ?", (record_id,))
            return cur.rowcount > 0

    def list_ingest_batches(self) -> list[sqlite3.Row]:
        with self._conn() as conn:
            return conn.execute(
                "SELECT ingest_batch, COUNT(*) AS note_count, MIN(created_at) AS created_at "
                "FROM processed_cards WHERE ingest_batch IS NOT NULL "
                "GROUP BY ingest_batch ORDER BY ingest_batch DESC"
            ).fetchall()

    def delete_ingest_batch(self, batch_id: str) -> tuple[int, int]:
        """§5: deshace una ingesta — borra sus registros NO subidos a Anki.

        Devuelve (borrados, conservados_por_estar_subidos). No borra los 'subida'
        para no orfanar notas en Anki (esos se revierten con undo de sync)."""
        with self._conn() as conn:
            kept = conn.execute(
                "SELECT COUNT(*) FROM processed_cards WHERE ingest_batch = ? AND status = ?",
                (batch_id, STATUS_UPLOADED),
            ).fetchone()[0]
            cur = conn.execute(
                "DELETE FROM processed_cards WHERE ingest_batch = ? AND status != ?",
                (batch_id, STATUS_UPLOADED),
            )
            return cur.rowcount, kept

    def stats(self) -> dict[str, int]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT action_taken, COUNT(*) as count FROM processed_cards GROUP BY action_taken"
            ).fetchall()
        return {row["action_taken"]: row["count"] for row in rows}

    def replace_indexed_notes(
        self,
        *,
        deck_name: str,
        include_subdecks: bool,
        notes: list[dict[str, Any]],
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            if include_subdecks:
                conn.execute(
                    "DELETE FROM indexed_notes WHERE deck_name = ? OR deck_name LIKE ?",
                    (deck_name, f"{deck_name}::%"),
                )
            else:
                conn.execute(
                    "DELETE FROM indexed_notes WHERE deck_name = ?",
                    (deck_name,),
                )

            if notes:
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO indexed_notes
                        (note_id, deck_name, note_type, front_original, back_original,
                         front_normalized, back_normalized, fingerprint,
                         scope_json,
                         intent, semantic_key, anchor_tokens, trigram_signature,
                         content_tokens, back_tokens, char_trigrams, block_keys,
                         embedding_vector, indexed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            note["note_id"],
                            note["deck_name"],
                            note["note_type"],
                            note["front_original"],
                            note["back_original"],
                            note["front_normalized"],
                            note["back_normalized"],
                            note["fingerprint"],
                            note.get("scope_json", "{}"),
                            note.get("intent", "unknown"),
                            note["semantic_key"],
                            note["anchor_tokens"],
                            note["trigram_signature"],
                            note.get("content_tokens", ""),
                            note.get("back_tokens", ""),
                            note.get("char_trigrams", ""),
                            note.get("block_keys", ""),
                            note.get("embedding_vector", ""),
                            note["indexed_at"],
                        )
                        for note in notes
                    ],
                )

            conn.execute(
                """
                INSERT OR REPLACE INTO indexed_decks
                    (deck_name, include_subdecks, last_refreshed_at, note_count)
                VALUES (?, ?, ?, ?)
                """,
                (deck_name, int(include_subdecks), now, len(notes)),
            )

    def list_indexed_notes(
        self,
        *,
        deck_name: str | None = None,
        include_subdecks: bool = True,
    ) -> list[sqlite3.Row]:
        with self._conn() as conn:
            if deck_name is None:
                rows = conn.execute(
                    "SELECT * FROM indexed_notes ORDER BY deck_name ASC, note_id ASC"
                ).fetchall()
            elif include_subdecks:
                rows = conn.execute(
                    """
                    SELECT * FROM indexed_notes
                    WHERE deck_name = ? OR deck_name LIKE ?
                    ORDER BY deck_name ASC, note_id ASC
                    """,
                    (deck_name, f"{deck_name}::%"),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM indexed_notes
                    WHERE deck_name = ?
                    ORDER BY note_id ASC
                    """,
                    (deck_name,),
                ).fetchall()
        return rows

    # --- Embedding cache (E1-3 / E1-4) ---

    def get_cached_embeddings(
        self, fingerprints: list[str], model: str
    ) -> dict[str, list[float]]:
        """Recupera vectores cacheados para los fingerprints dados bajo `model`."""
        if not fingerprints:
            return {}
        result: dict[str, list[float]] = {}
        unique = list(dict.fromkeys(fingerprints))
        with self._conn() as conn:
            for start in range(0, len(unique), 500):
                chunk = unique[start : start + 500]
                placeholders = ",".join("?" for _ in chunk)
                rows = conn.execute(
                    f"""
                    SELECT fingerprint, vector FROM embedding_cache
                    WHERE model = ? AND fingerprint IN ({placeholders})
                    """,
                    (model, *chunk),
                ).fetchall()
                for row in rows:
                    try:
                        parsed = json.loads(row["vector"])
                    except (json.JSONDecodeError, TypeError):
                        continue
                    if isinstance(parsed, list) and parsed:
                        result[row["fingerprint"]] = parsed
        return result

    def put_cached_embeddings(
        self, vectors: dict[str, list[float]], model: str
    ) -> None:
        """Persiste vectores nuevos por fingerprint bajo `model` (idempotente)."""
        if not vectors:
            return
        now = datetime.now(timezone.utc).isoformat()
        rows = [
            (fingerprint, model, len(vector), json.dumps(list(vector)), now)
            for fingerprint, vector in vectors.items()
            if vector
        ]
        if not rows:
            return
        with self._conn() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO embedding_cache
                    (fingerprint, model, dim, vector, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                rows,
            )


def embedding_cache_io(registry: "Registry | None", model: str):
    """Closures (lookup, store) para que enrich reuse el cache por fingerprint.

    Si no hay registry (modo sin persistencia) devuelve None/None y enrich
    simplemente computa sin cachear.
    """
    if registry is None:
        return None, None
    return (
        lambda fingerprints: registry.get_cached_embeddings(fingerprints, model),
        lambda vectors: registry.put_cached_embeddings(vectors, model),
    )
