"""MCP server para Anki Card Manager.

Expone las funciones del pipeline como tools que un agente (Claude Code, etc.)
puede descubrir y llamar directamente.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from mcp.server.fastmcp import FastMCP

from acm.anki.client import AnkiConnectClient, AnkiConnectError
from acm.anki.indexer import fetch_recent_notes
from acm.config import load_profile_taxonomy, load_settings, save_taxonomy
from acm.models import AuditDecision, CandidateCard, CardScope
from acm.pipeline.auditor import audit_batch, correct_record
from acm.pipeline.classifier import classify_fields, _parse_tag
from acm.pipeline.duplicate_audit import (
    build_duplicate_pool,
    collection_health,
    find_duplicate_clusters,
    find_similar_card as find_similar_card_records,
    suggest_taxonomy_for_deck,
)
from acm.pipeline.embeddings import embeddings_available
from acm.pipeline.normalizer import normalize_format, normalize_text
from acm.pipeline.quality import quality_flags, suggest_cloze
from acm.pipeline.similarity import (
    DuplicateMatch,
    build_record_from_fields,
    find_similar_records,
    make_fingerprint,
    serialize_cluster,
    serialize_match,
    serialize_metrics,
)
from acm.service import (
    backup_registry as _backup_registry,
    resolve_model_name as _resolve_model_name,
    resolve_record as _service_resolve,
    scope_from_row as _scope_from_row,
    sync_pending as _service_sync,
    try_anki_client as _try_anki_client,
    undo_batch as _service_undo,
)
from acm.store.registry import Registry

mcp = FastMCP(
    "anki-card-manager",
    instructions=(
        "Anki Card Manager — capa local entre Claude y Anki: deduplica "
        "(cross-deck, semántica por embeddings), clasifica y rutea tarjetas, y "
        "las sube via AnkiConnect. Flujo recomendado: 1) acm_annotate para anotar "
        "candidatas SIN subir (devuelve duplicados, mazo, tags y calidad) y "
        "mostrar al usuario una lista ya revisable; 2) acm_ingest para persistir; "
        "3) acm_resolve(id, approve|reject) sobre la cola (acm_review); 4) acm_sync "
        "para subir las aprobadas. acm_audit(deck, mode=duplicates|recent|untagged) "
        "audita mazos existentes. acm_apply_tags, acm_taxonomy(action), acm_stats. "
        "Economía de tokens: las tools resuelven local y solo devuelven el caso difícil."
    ),
)


def _setup(profile_name: str | None = None, *, include_registry: bool = True):
    settings = load_settings()
    registry = Registry(settings.db_path_resolved) if include_registry else None
    resolved_profile_name, profile, taxonomy = load_profile_taxonomy(settings, profile_name)
    return registry, settings, resolved_profile_name, profile, taxonomy


def _matches_from_row(row) -> list:
    """E1-5: detalle de duplicados persistido (score + razón + front/deck)."""
    raw = row["match_json"] if "match_json" in row.keys() else "[]"
    try:
        parsed = json.loads(raw or "[]")
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def _row_status(row) -> str | None:
    return row["status"] if "status" in row.keys() else None


def _compact_match(match: dict) -> dict:
    """§9: match minimal para la salida (sin repetir front/back/scope completos).

    §10: solo adjunta `mejor_version` cuando sugiere reemplazo (accionable); como
    los matches ya son duplicados reales (gating de precisión), es seguro."""
    out = {
        "id": match.get("id"),
        "deck": match.get("deck"),
        "score": match.get("score"),
        "reason_codes": match.get("reason_codes"),
    }
    mejor = match.get("mejor_version")
    if isinstance(mejor, dict) and mejor.get("suggestion") == "replace_old_with_new":
        out["mejor_version"] = mejor
    return out


def _matches_out(match_details: list, verbose: bool, limit: int = 2) -> list:
    """Detalle completo si verbose; si no, top-N compacto (economía de tokens, §9)."""
    if verbose:
        return match_details
    return [_compact_match(m) for m in match_details[:limit]]


def _load_structured_input(
    *,
    raw_json: str | None = None,
    file: str | None = None,
):
    if raw_json and file:
        raise ValueError("Usa solo uno: 'assignments_json' o 'file'")
    if file:
        input_path = Path(file).expanduser()
        if not input_path.exists():
            raise FileNotFoundError(f"No existe el archivo: {input_path}")
        contents = input_path.read_text(encoding="utf-8")
        if input_path.suffix.lower() in {".yaml", ".yml"}:
            return yaml.safe_load(contents)
        return json.loads(contents)
    if raw_json is None:
        raise ValueError("Debes enviar 'assignments_json' o 'file'")
    return json.loads(raw_json)


def _combined_categories(*category_groups: list[str]) -> list[str]:
    categories: list[str] = []
    for group in category_groups:
        for category in group:
            if category not in categories:
                categories.append(category)
    return categories


def _missing_categories(
    *,
    categories: list[str],
    existing_tags: set[str],
    classified,
) -> list[str]:
    missing: list[str] = []
    for category in categories:
        has_category = any(t.startswith(f"{category}::") for t in existing_tags) or any(
            t.startswith(f"{category}::") for t in classified.tags_resolved
        )
        if not has_category:
            missing.append(category)
    return missing


def _embeddings_used(settings) -> bool:
    """E0-3: ¿se usarán embeddings en esta corrida? Honesto sobre la degradación.

    False si están desactivados o si Ollama/el modelo no responden (degrada a
    léxico). Las tools lo exponen para que el agente sepa en qué modo corrió.
    """
    if not settings.acm.use_embeddings:
        return False
    return embeddings_available(
        ollama_url=settings.acm.ollama_url,
        model=settings.acm.ollama_model,
    )


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool()
def acm_ingest(cards_json: str, on_exact_match: str = "update", verbose: bool = False) -> str:
    """Procesa tarjetas candidatas: normaliza, clasifica, deduplica, decide y persiste.

    Args:
        cards_json: JSON array de tarjetas. Cada tarjeta tiene:
            - front (str, requerido): Pregunta de la tarjeta
            - back (str, requerido): Respuesta de la tarjeta
            - source (str, requerido): Origen ("claude", "chatgpt", "manual")
            - suggested_tags (list[str], opcional): Tags en formato "category::value"
            - note_type (str, opcional): Tipo de nota Anki (default "Basic")
        on_exact_match: qué hacer si el contenido ya existe (mismo fingerprint):
            "update" (default, actualiza el registro existente — idempotente),
            "skip" (no toca el existente) o "new" (crea uno nuevo). Evita duplicar
            el registro al re-ingerir (reporte §7).

    Returns:
        JSON con la decisión/persistencia por tarjeta (insert/possible_duplicate/
        reject/updated/skipped).
    """
    registry, settings, profile_name, profile, taxonomy = _setup()

    data = json.loads(cards_json)
    if not isinstance(data, list):
        return json.dumps({"error": "Input debe ser una lista de tarjetas"})

    cards: list[CandidateCard] = []
    parse_errors: list[str] = []
    for i, item in enumerate(data):
        try:
            cards.append(CandidateCard(**item))
        except Exception as e:
            parse_errors.append(f"Tarjeta {i}: {e}")

    if not cards:
        return json.dumps({"error": "Sin tarjetas válidas", "parse_errors": parse_errors})

    anki_client: AnkiConnectClient | None = None
    try:
        client = AnkiConnectClient(settings.anki.connect_url)
        if client.is_available():
            anki_client = client
    except Exception:
        pass

    decisions = audit_batch(
        cards,
        registry,
        taxonomy,
        settings,
        anki_client,
        profile_name=profile_name,
        profile=profile,
    )

    # Persistencia idempotente (§7): si el contenido ya existe (mismo
    # fingerprint), upsert en sitio en vez de duplicar el registro.
    mode = on_exact_match.strip().lower()
    if mode not in {"update", "skip", "new"}:
        mode = "update"

    persisted = []  # (decision, outcome, record_id)
    for d in decisions:
        existing = None if mode == "new" else registry.find_by_fingerprint(d.card.fingerprint)
        if existing is not None:
            if mode == "update":
                registry.update_card_fields(existing["id"], d.card)
                persisted.append((d, "updated", existing["id"]))
            else:  # skip
                persisted.append((d, "skipped", existing["id"]))
        else:
            record_id = registry.insert(d)
            persisted.append((d, d.action, record_id))

    if anki_client:
        anki_client.close()

    results = []
    for d, outcome, record_id in persisted:
        results.append({
            "front": d.card.front,
            "action": outcome,        # insert/possible_duplicate/reject/updated/skipped
            "audit": d.action,        # veredicto de auditoría (antes de upsert)
            "reason": d.reason,
            "id_short": record_id[:8],
            "vendor": d.card.scope.vendor,
            "topic": d.card.scope.topic,
            "scope": d.card.scope.summary(),
            "tags_resolved": d.card.tags_resolved,
            "tags_unresolved": d.card.tags_unresolved,
            "confianza": _classify_confidence(d.card, profile.required_categories),
            "matches": _matches_out(d.match_details, verbose),
            "profile": d.card.profile,
            "deck": d.card.deck,
        })

    summary = {}
    for _decision, outcome, _rid in persisted:
        summary[outcome] = summary.get(outcome, 0) + 1

    return json.dumps({
        "decisions": results,
        "summary": summary,
        "on_exact_match": mode,
        "anki_available": anki_client is not None,
        "embeddings_used": _embeddings_used(settings),
        "parse_errors": parse_errors,
    })


@mcp.tool()
def acm_annotate(cards_json: str, verbose: bool = False) -> str:
    """Anota tarjetas candidatas SIN subir ni persistir nada (E3-1 / RF-A2).

    Es el corazón del flujo "crear → revisar → subir": Claude propone cards y
    llama a esto; cada card vuelve anotada para que muestres al usuario una lista
    YA deduplicada y clasificada. No escribe en el registro ni en Anki.

    Por cada card devuelve:
      - es_duplicado (bool) + matches (id, deck, score, razón) cross-deck
      - mazo_sugerido, tags_sugeridos (alta confianza), tags_ambiguos
      - confianza (high/medium/low), flags_calidad, material_origen

    Args:
        cards_json: JSON array. Cada card: front, back, source, y opcionales
            suggested_tags ("category::value"), note_type, profile, deck,
            material_origen (PDF/sección de origen).
        verbose: False (default) → matches compactos (top-2) para economía de
            tokens (§9); True → detalle completo de cada match.
    """
    registry, settings, profile_name, profile, taxonomy = _setup()

    data = json.loads(cards_json)
    if not isinstance(data, list):
        return json.dumps({"error": "Input debe ser una lista de tarjetas"})

    cards: list[CandidateCard] = []
    parse_errors: list[str] = []
    for i, item in enumerate(data):
        try:
            cards.append(CandidateCard(**item))
        except Exception as e:
            parse_errors.append(f"Tarjeta {i}: {e}")
    if not cards:
        return json.dumps({"error": "Sin tarjetas válidas", "parse_errors": parse_errors})

    anki_client = _try_anki_client(settings)

    # audit_batch analiza (clasifica + dedup cross-deck + propaga tags) SIN
    # persistir — la persistencia es responsabilidad de acm_ingest, no de anotar.
    decisions = audit_batch(
        cards, registry, taxonomy, settings, anki_client,
        profile_name=profile_name, profile=profile,
    )

    # Validación temprana de note_type contra Anki (reporte §4): avisar AHORA,
    # no tras 12 aprobaciones y un sync fallido.
    available_models = anki_client.get_model_names() if anki_client else []

    annotations = []
    for decision in decisions:
        card = decision.card
        deck = card.deck
        if deck is None and anki_client is not None and profile.root_deck:
            deck = anki_client.resolve_deck(
                scope=card.scope,
                root_deck=profile.root_deck,
                routing_categories=profile.routing_categories,
            )
        clean_front = normalize_format(card.front)
        clean_back = normalize_format(card.back)
        formato_sugerido = {}
        if clean_front != card.front:
            formato_sugerido["front"] = clean_front
        if clean_back != card.back:
            formato_sugerido["back"] = clean_back

        note_type_warning = None
        if available_models:
            _resolved, _model_err = _resolve_model_name(card.note_type, settings, available_models)
            if _model_err:
                note_type_warning = _model_err  # p.ej. "el modelo 'Basic' no existe… ¿Básico?"

        annotations.append({
            "front": card.front,
            "es_duplicado": decision.action == "possible_duplicate",
            "action": decision.action,
            "matches": _matches_out(decision.match_details, verbose),
            "mazo_sugerido": deck,
            "tags_sugeridos": card.tags_resolved,
            "tags_ambiguos": card.tags_unresolved,
            "scope": card.scope.summary(),
            "confianza": _classify_confidence(card, profile.required_categories),
            "flags_calidad": quality_flags(card.front, card.back),
            "sugerencia_cloze": suggest_cloze(card.front, card.back),  # E8-3
            "formato_sugerido": formato_sugerido or None,  # E8-4
            "note_type_warning": note_type_warning,  # §4: aviso temprano de modelo
            "material_origen": card.material_origen,
        })

    if anki_client:
        anki_client.close()

    summary = {
        "total": len(annotations),
        "duplicados": sum(1 for a in annotations if a["es_duplicado"]),
        "alta_confianza": sum(1 for a in annotations if a["confianza"] == "high"),
        "con_flags_calidad": sum(1 for a in annotations if a["flags_calidad"]),
    }
    return json.dumps({
        "annotations": annotations,
        "summary": summary,
        "anki_available": anki_client is not None,
        "embeddings_used": _embeddings_used(settings),
        "parse_errors": parse_errors,
    })


@mcp.tool()
def acm_review() -> str:
    """Lista tarjetas marcadas como possible_duplicate pendientes de revisión.

    Returns:
        JSON con la lista de tarjetas pendientes, cada una con id, front, vendor, topic y fecha.
    """
    registry, _, _, _, _ = _setup()
    rows = registry.list_pending_review()

    return json.dumps({
        "count": len(rows),
        "cards": [
            {
                "id": row["id"],
                "id_short": row["id"][:8],
                "front": row["front_original"],
                "front_normalized": row["front_normalized"],
                "scope": _scope_from_row(row).summary(),
                "profile": row["profile_name"] if "profile_name" in row.keys() else None,
                "deck": row["target_deck"] if "target_deck" in row.keys() else None,
                "estado": _row_status(row),
                "created_at": row["created_at"],
                "matches": _matches_from_row(row),
            }
            for row in rows
        ],
    })


@mcp.tool()
def acm_resolve(
    record_id: str,
    action: str,
    front: str | None = None,
    back: str | None = None,
    tags: list[str] | None = None,
    note_type: str | None = None,
    deck: str | None = None,
) -> str:
    """Resuelve un item de la cola de revisión con UNA sola acción (E5-1 / E5-3).

    Funciona en CUALQUIER estado (id completo o prefijo), no solo en la cola
    activa (§6). La cola incluye duplicados posibles y clasificación ambigua.

    Args:
        record_id: ID completo o prefijo del registro.
        action: "approve" (→ aprobada), "reject" (→ descartada), o "correct".
        front, back: contenido corregido (requeridos para action="correct").
        tags: tags sugeridos para la corrección (opcional, "category::value").
        note_type, deck: corrección opcional de modelo/mazo (§5).

    Para "correct", la card corregida REINGRESA al pipeline: se re-deduplica y
    re-clasifica automáticamente (E5-3).
    """
    normalized = action.strip().lower()

    if normalized in {"approve", "reject"}:
        registry, _, _, _, _ = _setup()
        return json.dumps(_service_resolve(registry, record_id, normalized))

    if normalized == "correct":
        if not front or not back:
            return json.dumps({"error": "action='correct' requiere 'front' y 'back'"})
        registry, settings, profile_name, profile, taxonomy = _setup()
        anki_client = _try_anki_client(settings)
        decision = correct_record(
            record_id, front=front, back=back, tags=tags or [],
            registry=registry, taxonomy=taxonomy, settings=settings,
            profile_name=profile_name, profile=profile,
            note_type=note_type, deck=deck, anki_client=anki_client,
        )
        if anki_client:
            anki_client.close()
        if decision is None:
            return json.dumps({"error": f"No se encontró registro: {record_id}"})
        row = registry.get_by_id(record_id)
        return json.dumps({
            "status": "corrected", "id": record_id, "action": decision.action,
            "estado": _row_status(row),
            "es_duplicado": decision.action == "possible_duplicate",
            "tags_resolved": decision.card.tags_resolved,
            "matches": decision.match_details,
        })

    return json.dumps({"error": "action debe ser 'approve', 'reject' o 'correct'"})


@mcp.tool()
def acm_sync(export_tsv: str | None = None, dry_run: bool = False) -> str:
    """Empuja las tarjetas aprobadas pendientes a Anki, ruteadas e idempotente.

    Re-correr no recrea: las subidas pasan a estado 'subida' y se saltan (E4-3).
    Anki cerrado → las aprobadas quedan encoladas y suben al reconectar; con
    `export_tsv` (ruta) además se exportan a TSV (E4-5). `dry_run=True` muestra el
    plan sin tocar Anki (E9-1). Antes de subir hace backup del registro (E9-3) y
    devuelve un `batch_id` deshacible con acm_undo (E9-2).

    Returns:
        JSON con synced (id/deck/note_id), errores, batch_id y backup.
    """
    registry, settings, _, _, _ = _setup()
    path = Path(export_tsv).expanduser() if export_tsv else None
    result = _service_sync(registry, settings, export_tsv_path=path, dry_run=dry_run)
    result.setdefault("embeddings_used", False)
    return json.dumps(result)


@mcp.tool()
def acm_undo(batch_id: str) -> str:
    """E9-2: deshace un lote de sync — borra esas notas de Anki y revierte el
    estado de las tarjetas a 'aprobada'. Mirá los lotes con acm_stats.

    Args:
        batch_id: id del lote (lo devuelve acm_sync y lo lista acm_stats).
    """
    registry, settings, _, _, _ = _setup()
    return json.dumps(_service_undo(registry, settings, batch_id))


@mcp.tool()
def acm_audit(
    deck: str,
    mode: str = "duplicates",
    profile: str | None = None,
    include_subdecks: bool = True,
    days: int = 1,
    include_registry: bool = True,
    limit: int = 50,
    offset: int = 0,
) -> str:
    """Audita un deck de Anki — una sola tool para tres modos (E6-2).

    Modos:
      - "duplicates": clusters de tarjetas repetidas (dedup local kNN cross-deck).
      - "recent": audita lo creado en Anki en los últimos `days` días (dups +
        tags faltantes). Para la pasada de "lo que hiciste a mano" (RF-F1).
      - "untagged": notas sin tags de taxonomía completos; auto-taggea las de
        alta confianza y devuelve solo las ambiguas, paginadas.

    Args:
        deck: deck raíz a auditar.
        mode: "duplicates" | "recent" | "untagged".
        profile: perfil de taxonomía (default: el de settings).
        include_subdecks: incluir subdecks (default True).
        days: ventana temporal para mode="recent".
        include_registry: incluir el registro local en mode="duplicates".
        limit, offset: paginación para mode="untagged".
    """
    normalized = mode.strip().lower()
    if normalized == "duplicates":
        return acm_find_duplicates(
            deck, profile=profile, include_registry=include_registry,
            include_subdecks=include_subdecks,
        )
    if normalized == "recent":
        return acm_audit_recent(deck, days=days, profile=profile, include_subdecks=include_subdecks)
    if normalized == "untagged":
        return acm_list_untagged(
            deck, profile=profile, include_subdecks=include_subdecks, limit=limit, offset=offset,
        )
    if normalized == "suggest_taxonomy":
        return acm_suggest_taxonomy(deck, profile=profile, include_subdecks=include_subdecks)
    if normalized == "maintenance":
        return acm_maintenance(deck, profile=profile, include_subdecks=include_subdecks)
    return json.dumps({
        "error": f"mode inválido: {mode}. Usa duplicates|recent|untagged|suggest_taxonomy|maintenance"
    })


def acm_maintenance(deck: str, profile: str | None = None, include_subdecks: bool = True) -> str:
    """E7-2: reporte de mantenimiento read-only (dups + huérfanas + leeches).
    Solo reporta/propone; no toca nada (vía acm_audit mode=maintenance)."""
    registry, settings, profile_name, profile_config, taxonomy = _setup(profile)
    anki_client = _try_anki_client(settings)
    report = collection_health(
        registry=registry, settings=settings, taxonomy=taxonomy,
        profile_name=profile_name, profile=profile_config,
        deck=deck, include_subdecks=include_subdecks, anki_client=anki_client,
    )
    if anki_client:
        anki_client.close()
    return json.dumps({
        "deck": deck,
        **report,
        "anki_available": anki_client is not None,
        "embeddings_used": _embeddings_used(settings),
    })


@mcp.tool()
def acm_reorganize(
    deck: str,
    profile: str | None = None,
    include_subdecks: bool = True,
    dry_run: bool = True,
) -> str:
    """E7-3: reorganización masiva (one-shot) de un mazo — re-taggea y reporta dups.

    `dry_run=True` (default) previsualiza SIN tocar nada (E9-1). Al aplicar, hace
    backup del registro (E9-3), auto-taggea en Anki las notas de alta confianza y
    reporta los duplicados para revisión (NUNCA fusiona; decisión #2).

    Args:
        deck: deck raíz a reorganizar.
        profile, include_subdecks: comunes.
        dry_run: True previsualiza; False aplica (con backup previo).
    """
    registry, settings, profile_name, profile_config, taxonomy = _setup(profile)

    if dry_run:
        anki_client = _try_anki_client(settings)
        plan = collection_health(
            registry=registry, settings=settings, taxonomy=taxonomy,
            profile_name=profile_name, profile=profile_config,
            deck=deck, include_subdecks=include_subdecks, anki_client=anki_client,
        )
        if anki_client:
            anki_client.close()
        return json.dumps({
            "dry_run": True, "deck": deck, "plan": plan,
            "note": "Pasá dry_run=false para aplicar (auto-tag alta confianza + backup previo).",
            "anki_available": anki_client is not None,
            "embeddings_used": _embeddings_used(settings),
        })

    # Aplicar: backup → auto-tag (deck-wide) → reportar dups (sin fusionar).
    backup_path = _backup_registry(settings)
    tagged = json.loads(acm_list_untagged(
        deck, profile=profile, include_subdecks=include_subdecks,
        limit=100000, include_auto_tagged=True,
    ))
    dups = json.loads(acm_find_duplicates(deck, profile=profile, include_subdecks=include_subdecks))
    summary = tagged.get("summary", {}) if isinstance(tagged, dict) else {}
    return json.dumps({
        "dry_run": False, "deck": deck, "backup": str(backup_path),
        "auto_tagged": summary.get("auto_tagged", 0),
        "still_needs_review": summary.get("needs_review", 0),
        "duplicate_clusters": len(dups.get("clusters", [])) if isinstance(dups, dict) else 0,
        "embeddings_used": _embeddings_used(settings),
    })


def acm_suggest_taxonomy(deck: str, profile: str | None = None, include_subdecks: bool = True) -> str:
    """E2-6: propone tags nuevos para clusters de cards sin clasificar (vía acm_audit
    mode=suggest_taxonomy). Solo propone; nunca modifica la taxonomía."""
    registry, settings, profile_name, profile_config, taxonomy = _setup(profile)
    anki_client = _try_anki_client(settings)
    proposals = suggest_taxonomy_for_deck(
        registry=registry, settings=settings, taxonomy=taxonomy,
        profile_name=profile_name, profile=profile_config,
        deck=deck, include_subdecks=include_subdecks, anki_client=anki_client,
    )
    if anki_client:
        anki_client.close()
    return json.dumps({
        "deck": deck,
        "proposals": proposals,
        "anki_available": anki_client is not None,
        "embeddings_used": _embeddings_used(settings),
        "summary": {
            "proposals": len(proposals),
            "new_tags": sum(1 for p in proposals if p["is_new"]),
        },
    })


def acm_find_duplicates(
    deck: str,
    profile: str | None = None,
    include_registry: bool = True,
    include_subdecks: bool = True,
) -> str:
    """Busca clusters locales de tarjetas repetidas para un deck (vía acm_audit mode=duplicates)."""
    registry, settings, profile_name, profile, taxonomy = _setup(profile)

    anki_client: AnkiConnectClient | None = None
    try:
        client = AnkiConnectClient(settings.anki.connect_url)
        if client.is_available():
            anki_client = client
    except Exception:
        anki_client = None

    if not anki_client and not registry.list_indexed_notes(deck_name=deck, include_subdecks=include_subdecks):
        return json.dumps({
            "error": "Anki no disponible y no hay índice local para ese deck",
            "deck": deck,
        })

    clusters, metrics = find_duplicate_clusters(
        registry=registry,
        settings=settings,
        taxonomy=taxonomy,
        profile_name=profile_name,
        profile=profile,
        deck=deck,
        include_subdecks=include_subdecks,
        include_registry=include_registry,
        anki_client=anki_client,
        refresh_index=anki_client is not None,
    )

    if anki_client:
        anki_client.close()

    return json.dumps({
        "deck": deck,
        "include_registry": include_registry,
        "include_subdecks": include_subdecks,
        "anki_available": anki_client is not None,
        "embeddings_used": _embeddings_used(settings),
        "clusters": [serialize_cluster(cluster) for cluster in clusters],
        "metrics": serialize_metrics(metrics),
    })


@mcp.tool()
def acm_find_similar_card(
    front: str,
    back: str = "",
    deck: str | None = None,
    profile: str | None = None,
) -> str:
    """Busca tarjetas similares a un query usando el motor local de duplicados."""
    registry, settings, profile_name, profile, taxonomy = _setup(profile)

    anki_client: AnkiConnectClient | None = None
    try:
        client = AnkiConnectClient(settings.anki.connect_url)
        if client.is_available():
            anki_client = client
    except Exception:
        anki_client = None

    if deck and not anki_client and not registry.list_indexed_notes(deck_name=deck, include_subdecks=settings.acm.audit_include_subdecks):
        return json.dumps({
            "error": "Anki no disponible y no hay índice local para ese deck",
            "deck": deck,
        })

    matches = find_similar_card_records(
        registry=registry,
        settings=settings,
        taxonomy=taxonomy,
        profile_name=profile_name,
        profile=profile,
        front=front,
        back=back,
        note_type=settings.anki.default_model or "Basic",
        deck=deck,
        include_subdecks=settings.acm.audit_include_subdecks,
        anki_client=anki_client,
    )

    if anki_client:
        anki_client.close()

    return json.dumps({
        "matches": [serialize_match(match) for match in matches],
        "anki_available": anki_client is not None,
        "embeddings_used": _embeddings_used(settings),
    })


@mcp.tool()
def acm_stats() -> str:
    """Muestra estadísticas del registro de tarjetas procesadas.

    Returns:
        JSON con conteo por acción (insert, possible_duplicate, reject) y total.
    """
    registry, _, _, _, _ = _setup()
    data = registry.stats()
    metrics = registry.metrics()
    batches = [
        {"batch_id": row["sync_batch"], "note_count": row["note_count"],
         "created_at": row["created_at"]}
        for row in registry.list_sync_batches()
    ]
    return json.dumps({
        "by_action": data,
        "total": sum(data.values()),
        # E9-4: observabilidad — estados + auto-resueltas vs escaladas al usuario.
        "metrics": metrics,
        "sync_batches": batches,  # E9-2: lotes deshacibles con acm_undo
    })


@mcp.tool()
def acm_taxonomy(
    action: str = "show",
    category: str | None = None,
    value: str | None = None,
    profile: str | None = None,
) -> str:
    """Consulta o edita la taxonomía de tags (E6-2: show + add en una tool).

    Args:
        action: "show" (default, lista categorías y valores) o "add".
        category: categoría del tag (requerido para action="add").
        value: valor a agregar (requerido para action="add"); crea category::value.
        profile: perfil cuya taxonomía consultar/editar.

    Returns:
        JSON con la taxonomía (show) o el resultado de la operación (add).
    """
    _, settings, _, profile_config, taxonomy = _setup(profile, include_registry=False)
    act = action.strip().lower()

    if act == "show":
        return json.dumps(taxonomy.model_dump())

    if act == "add":
        if not category or not value:
            return json.dumps({"error": "action='add' requiere 'category' y 'value'"})
        if value in taxonomy.values_for(category):
            return json.dumps({"status": "exists", "message": f"'{value}' ya existe en '{category}'"})
        taxonomy.append_value(category, value)
        save_taxonomy(taxonomy, profile_config.taxonomy_path_resolved(settings.taxonomy_path_resolved))
        return json.dumps({"status": "added", "tag": f"{category}::{value}"})

    return json.dumps({"error": f"action inválida: {action}. Usa show|add"})


def _classify_confidence(classified, required_categories: list[str] | None = None) -> str:
    """Determina si la clasificación es confiable o ambigua."""
    required_categories = required_categories or ["vendor", "topic"]
    has_unresolved = len(classified.tags_unresolved) > 0
    resolved_count = len(classified.tags_resolved)
    covered_required = [
        category
        for category in required_categories
        if classified.scope.get(category) is not None
        or any(tag.startswith(f"{category}::") for tag in classified.tags_resolved)
    ]

    if required_categories and len(covered_required) == len(required_categories) and not has_unresolved:
        return "high"
    if covered_required or (resolved_count >= 1 and not has_unresolved):
        return "medium"
    return "low"


@mcp.tool()
def acm_auto_classify(cards_json: str) -> str:
    """Clasifica tarjetas determinísticamente sin usar LLM. Retorna las que
    se pudieron resolver y las ambiguas que necesitan intervención del agente.

    Flujo token-eficiente: llama esto primero. Solo las cards en "needs_review"
    requieren que el agente las analice y llame acm_apply_tags.

    Args:
        cards_json: JSON array de tarjetas. Cada tarjeta tiene:
            - front (str): Pregunta
            - back (str): Respuesta
            - source (str, opcional): Origen (default "manual")
            - suggested_tags (list[str], opcional): Tags existentes "category::value"
            - note_type (str, opcional): Tipo de nota Anki (default "Basic")

    Returns:
        JSON con:
        - classified: tarjetas resueltas con alta confianza (no requieren tokens)
        - needs_review: tarjetas ambiguas con contexto para que el agente decida
        - taxonomy: categorías válidas (solo si hay items en needs_review)
        - summary: conteos
    """
    _, settings, default_profile_name, default_profile, taxonomy = _setup(include_registry=False)
    profile_cache: dict[str, tuple[object, object]] = {
        default_profile_name: (default_profile, taxonomy)
    }
    taxonomies_for_review: dict[str, dict] = {}

    data = json.loads(cards_json)
    if not isinstance(data, list):
        return json.dumps({"error": "Input debe ser una lista de tarjetas"})

    classified_cards = []
    needs_review = []

    for i, item in enumerate(data):
        try:
            front = item["front"]
            back = item.get("back", "")
            source = item.get("source", "manual")
            suggested_tags = item.get("suggested_tags", [])
            note_type = item.get("note_type", "Basic")
            requested_profile = item.get("profile") or default_profile_name
            deck = item.get("deck")

            if requested_profile in profile_cache:
                resolved_profile, resolved_taxonomy = profile_cache[requested_profile]
            else:
                _, resolved_profile, resolved_taxonomy = load_profile_taxonomy(settings, requested_profile)
                profile_cache[requested_profile] = (resolved_profile, resolved_taxonomy)

            fp = make_fingerprint(normalize_text(front), normalize_text(back))
            classified = classify_fields(
                front=front,
                back=back,
                source=source,
                suggested_tags=suggested_tags,
                note_type=note_type,
                taxonomy=resolved_taxonomy,
                fingerprint=fp,
                profile_name=requested_profile,
                profile=resolved_profile,
                deck=deck,
            )

            confidence = _classify_confidence(classified, resolved_profile.required_categories)

            card_result = {
                "index": i,
                "front": front,
                "back": back,
                "source": source,
                "note_type": note_type,
                "vendor": classified.scope.vendor,
                "topic": classified.scope.topic,
                "cert": classified.scope.cert,
                "type": classified.scope.get("type"),
                "scope": classified.scope.summary(),
                "tags_resolved": classified.tags_resolved,
                "tags_unresolved": classified.tags_unresolved,
                "confidence": confidence,
                "fingerprint": fp,
                "profile": requested_profile,
                "deck": deck,
            }

            if confidence == "high":
                classified_cards.append(card_result)
            else:
                # Para ambiguas, incluir hints del clasificador
                card_result["hints"] = {
                    "detected_scope": classified.scope.summary(),
                    "detected_vendor": classified.scope.vendor,
                    "detected_topic": classified.scope.topic,
                    "detected_type": classified.scope.get("type"),
                    "missing": [],
                }
                for category in resolved_profile.required_categories:
                    has_category = classified.scope.get(category) is not None or any(
                        t.startswith(f"{category}::") for t in classified.tags_resolved
                    )
                    if not has_category:
                        card_result["hints"]["missing"].append(category)
                needs_review.append(card_result)
                taxonomies_for_review[requested_profile] = resolved_taxonomy.model_dump()

        except Exception as e:
            needs_review.append({
                "index": i,
                "front": item.get("front", "?"),
                "back": item.get("back", ""),
                "error": str(e),
                "confidence": "error",
                "hints": {"missing": []},
            })

    result: dict = {
        "classified": classified_cards,
        "needs_review": needs_review,
        "embeddings_used": False,  # ruta determinista, sin Ollama
        "summary": {
            "total": len(data),
            "auto_classified": len(classified_cards),
            "needs_review": len(needs_review),
            "tokens_saved_pct": round(len(classified_cards) / max(len(data), 1) * 100),
        },
    }

    # Solo incluir taxonomía si hay items que revisar (ahorro de tokens)
    if needs_review:
        if len(taxonomies_for_review) == 1:
            result["taxonomy"] = next(iter(taxonomies_for_review.values()))
        else:
            result["taxonomies"] = taxonomies_for_review

    return json.dumps(result)


@mcp.tool()
def acm_apply_tags(assignments_json: str | None = None, file: str | None = None) -> str:
    """Aplica tags decididos por el agente o humano a tarjetas.

    Úsalo después de acm_auto_classify para las tarjetas en needs_review.
    Puede aplicar tags tanto a tarjetas nuevas (por índice del batch) como a
    notas existentes en Anki (por note_id).

    Args:
        assignments_json: JSON array de asignaciones. Cada una tiene:
            - tags (list[str]): Tags en formato "category::value"
            Y uno de:
            - note_id (int): Para notas ya existentes en Anki
            - card (dict): Para tarjetas nuevas, con front/back/source
        file: Ruta opcional a un archivo JSON/YAML con las asignaciones.

    Returns:
        JSON con resultados: tags aplicados, errores de validación, tarjetas
        procesadas vía el pipeline normal de ingest.
    """
    registry, settings, profile_name, profile, taxonomy = _setup()
    valid_tags = taxonomy.all_valid_values()

    try:
        data = _load_structured_input(raw_json=assignments_json, file=file)
    except (ValueError, FileNotFoundError, json.JSONDecodeError, yaml.YAMLError) as e:
        return json.dumps({"error": str(e)})

    if not isinstance(data, list):
        return json.dumps({"error": "Input debe ser una lista de asignaciones"})

    anki_client: AnkiConnectClient | None = None
    try:
        client = AnkiConnectClient(settings.anki.connect_url)
        if client.is_available():
            anki_client = client
    except Exception:
        pass

    results_anki = []
    cards_to_ingest: list[CandidateCard] = []
    errors = []

    for i, assignment in enumerate(data):
        tags = assignment.get("tags", [])

        # Validar tags contra taxonomía
        resolved = [t for t in tags if t in valid_tags]
        unresolved = [t for t in tags if t not in valid_tags and _parse_tag(t) is not None]
        invalid = [t for t in tags if _parse_tag(t) is None]

        if invalid:
            errors.append({
                "index": i,
                "error": f"Tags con formato inválido (usar category::value): {invalid}",
            })
            continue

        if "note_id" in assignment:
            # Aplicar tags a nota existente en Anki
            note_id = assignment["note_id"]
            if not anki_client:
                errors.append({"index": i, "note_id": note_id, "error": "Anki no disponible"})
                continue

            try:
                tags_str = " ".join(resolved)
                if tags_str:
                    anki_client.add_tags([note_id], tags_str)
                results_anki.append({
                    "note_id": note_id,
                    "tags_applied": resolved,
                    "tags_unresolved": unresolved,
                })
            except AnkiConnectError as e:
                errors.append({"index": i, "note_id": note_id, "error": str(e)})

        elif "card" in assignment:
            # Tarjeta nueva: agregar tags y enviar al pipeline de ingest
            card_data = assignment["card"]
            card_data["suggested_tags"] = tags
            try:
                cards_to_ingest.append(CandidateCard(**card_data))
            except Exception as e:
                errors.append({"index": i, "error": f"Card inválida: {e}"})
        else:
            errors.append({"index": i, "error": "Debe incluir 'note_id' o 'card'"})

    # Procesar tarjetas nuevas por el pipeline normal
    ingested = []
    if cards_to_ingest:
        decisions = audit_batch(
            cards_to_ingest,
            registry,
            taxonomy,
            settings,
            anki_client,
            profile_name=profile_name,
            profile=profile,
        )
        for d in decisions:
            registry.insert(d)
            ingested.append({
                "front": d.card.front,
                "action": d.action,
                "reason": d.reason,
                "tags_resolved": d.card.tags_resolved,
                "vendor": d.card.scope.vendor,
                "topic": d.card.scope.topic,
                "type": d.card.scope.get("type"),
                "scope": d.card.scope.summary(),
                "profile": d.card.profile,
                "deck": d.card.deck,
            })

    if anki_client:
        anki_client.close()

    return json.dumps({
        "anki_tags_applied": results_anki,
        "cards_ingested": ingested,
        "errors": errors,
        "anki_available": anki_client is not None,
        "embeddings_used": _embeddings_used(settings) if cards_to_ingest else False,
        "summary": {
            "anki_updated": len(results_anki),
            "cards_ingested": len(ingested),
            "errors": len(errors),
        },
    })


def acm_list_untagged(
    deck: str,
    profile: str | None = None,
    include_subdecks: bool = True,
    limit: int = 50,
    offset: int = 0,
    include_auto_tagged: bool = False,
) -> str:
    """Lista tarjetas de Anki que no tienen tags de taxonomía completos (vía acm_audit mode=untagged).

    Auto-clasifica lo que puede determinísticamente. Solo retorna las
    que necesitan intervención del agente, minimizando tokens.

    Args:
        deck: Nombre del deck a escanear (ej: "Cloud Certs").
        include_subdecks: Si incluir subdecks (default True).
        limit: Máximo de notas ambiguas a devolver por página.
        offset: Desplazamiento para paginar notas ambiguas.
        include_auto_tagged: Si incluir el detalle completo de las notas auto-etiquetadas.

    Returns:
        JSON con:
        - auto_tagged: notas que se pudieron clasificar (solo si include_auto_tagged=True)
        - needs_review: notas que necesitan que el agente decida los tags
        - taxonomy: categorías válidas (solo si hay items en needs_review)
        - summary: conteos
    """
    if limit <= 0:
        return json.dumps({"error": "limit debe ser mayor que 0"})
    if offset < 0:
        return json.dumps({"error": "offset no puede ser negativo"})

    _, settings, profile_name, profile_config, taxonomy = _setup(profile, include_registry=False)

    try:
        client = AnkiConnectClient(settings.anki.connect_url)
        if not client.is_available():
            return json.dumps({"error": "Anki no disponible. Abre Anki con AnkiConnect."})
    except Exception as e:
        return json.dumps({"error": f"No se pudo conectar: {e}"})

    valid_tags = taxonomy.all_valid_values()
    fm = settings.anki.field_mapping
    target_categories = _combined_categories(
        profile_config.required_categories,
        profile_config.routing_categories,
    )

    # Buscar notas en el deck — construir mapeo note_id → deck real
    # (notesInfo de AnkiConnect no incluye deckName; lo rastreamos aquí)
    decks = client.expand_decks(deck, include_subdecks)
    note_to_deck: dict[int, str] = {}
    all_note_ids: list[int] = []
    for d in decks:
        note_ids = client.find_notes(f'"deck:{d}"')
        for nid in note_ids:
            if nid not in note_to_deck:
                note_to_deck[nid] = d
        all_note_ids.extend(nid for nid in note_ids if nid not in note_to_deck or note_to_deck[nid] == d)

    # Deduplicar note_ids preservando orden
    seen_ids: set[int] = set()
    unique_note_ids: list[int] = []
    for nid in all_note_ids:
        if nid not in seen_ids:
            seen_ids.add(nid)
            unique_note_ids.append(nid)
    all_note_ids = unique_note_ids

    if not all_note_ids:
        client.close()
        return json.dumps({"error": f"No se encontraron notas en '{deck}'"})

    # Obtener info de todas las notas
    notes_info = client.get_notes_info(all_note_ids)

    auto_tagged = []
    needs_review = []

    for note in notes_info:
        note_id = note["noteId"]
        existing_tags = set(note.get("tags", []))

        missing_target = [
            category
            for category in target_categories
            if not any(t.startswith(f"{category}::") for t in existing_tags)
        ]

        if not missing_target:
            continue  # Ya está bien taggeada

        # Extraer front/back
        fields = note.get("fields", {})
        front = fields.get(fm.front, {}).get("value", "")
        back = fields.get(fm.back, {}).get("value", "")
        note_deck = note_to_deck.get(note_id) or note.get("deckName") or deck

        if not front:
            continue

        # Intentar clasificar determinísticamente
        taxonomy_tags = [t for t in existing_tags if t in valid_tags]
        fp = make_fingerprint(normalize_text(front), normalize_text(back))
        classified = classify_fields(
            front=front,
            back=back,
            source="anki",
            suggested_tags=taxonomy_tags,
            note_type=note.get("modelName", "Basic"),
            taxonomy=taxonomy,
            fingerprint=fp,
            profile_name=profile_name,
            profile=profile_config,
            deck=note_deck,
        )

        confidence = _classify_confidence(classified, profile_config.required_categories)

        # Construir los tags nuevos que se pueden agregar
        new_tags = [t for t in classified.tags_resolved if t not in existing_tags]
        missing_after_classification = _missing_categories(
            categories=target_categories,
            existing_tags=existing_tags,
            classified=classified,
        )

        if confidence == "high" and new_tags:
            # Aplicar automáticamente
            try:
                client.add_tags([note_id], " ".join(new_tags))
                auto_tagged.append({
                    "note_id": note_id,
                    "front": front,
                    "tags_added": new_tags,
                    "deck": note_deck,
                })
            except AnkiConnectError as e:
                needs_review.append({
                    "note_id": note_id,
                    "front": front,
                    "deck": note_deck,
                    "missing": missing_after_classification,
                    "hints": classified.scope.summary(),
                    "error": str(e),
                })
        else:
            needs_review.append({
                "note_id": note_id,
                "front": front,
                "deck": note_deck,
                "missing": missing_after_classification,
                "hints": classified.scope.summary(),
            })

    client.close()

    total_needs_review = len(needs_review)
    paged_needs_review = needs_review[offset:offset + limit]

    result: dict = {
        "needs_review": paged_needs_review,
        "anki_available": True,
        "embeddings_used": False,
        "pagination": {
            "limit": limit,
            "offset": offset,
            "returned": len(paged_needs_review),
            "total": total_needs_review,
            "has_more": offset + len(paged_needs_review) < total_needs_review,
            "next_offset": offset + len(paged_needs_review)
            if offset + len(paged_needs_review) < total_needs_review
            else None,
        },
        "summary": {
            "total_scanned": len(notes_info),
            "already_tagged": len(notes_info) - len(auto_tagged) - total_needs_review,
            "auto_tagged": len(auto_tagged),
            "needs_review": total_needs_review,
            "tokens_saved_pct": round(
                (len(notes_info) - total_needs_review) / max(len(notes_info), 1) * 100
            ),
        },
    }

    if include_auto_tagged:
        result["auto_tagged"] = auto_tagged

    if paged_needs_review:
        result["taxonomy"] = taxonomy.model_dump()

    return json.dumps(result)


def acm_audit_recent(
    deck: str,
    days: int = 1,
    profile: str | None = None,
    include_subdecks: bool = True,
) -> str:
    """Audita tarjetas agregadas recientemente en Anki directo desde la UI (vía acm_audit mode=recent).

    Diseñado para el flujo: el usuario crea cards en Anki y luego pide
    auditoría. Compara cards recientes contra todo el deck buscando
    duplicados, tags faltantes y problemas de calidad. Zero tokens para
    la creación — solo se gastan tokens al auditar.

    Args:
        deck: Nombre del deck a auditar (ej: "Cloud Certs").
        days: Ventana de tiempo en días (default 1 = hoy).
        profile: Perfil de taxonomía a usar (default: el del settings).
        include_subdecks: Si incluir subdecks (default True).

    Returns:
        JSON con:
        - recent_count: tarjetas encontradas en la ventana
        - duplicates: tarjetas con duplicados detectados
        - missing_tags: tarjetas con tags incompletos
        - ok: tarjetas sin problemas
        - metrics: estadísticas de comparación (reduction_pct)
    """
    registry, settings, profile_name, profile_config, taxonomy = _setup(profile)

    try:
        client = AnkiConnectClient(settings.anki.connect_url)
        if not client.is_available():
            return json.dumps({"error": "Anki no disponible. Abre Anki con AnkiConnect."})
    except Exception as e:
        return json.dumps({"error": f"No se pudo conectar: {e}"})

    # 1. Fetch recent notes
    recent_notes = fetch_recent_notes(
        client,
        deck=deck,
        include_subdecks=include_subdecks,
        days=days,
        settings=settings,
    )

    if not recent_notes:
        client.close()
        return json.dumps({
            "recent_count": 0,
            "message": f"Sin tarjetas nuevas en '{deck}' en los últimos {days} día(s)",
        })

    # 2. Build pool of ALL existing cards for comparison (uses blocking, not O(n²))
    pool = build_duplicate_pool(
        registry=registry,
        settings=settings,
        taxonomy=taxonomy,
        profile_name=profile_name,
        profile=profile_config,
        deck=deck,
        include_subdecks=include_subdecks,
        include_registry=True,
        anki_client=client,
        refresh_index=True,
        persist_index=True,
    )

    fm = settings.anki.field_mapping
    valid_tags = taxonomy.all_valid_values()
    required_categories = _combined_categories(
        profile_config.required_categories,
        profile_config.routing_categories,
    )

    duplicates = []
    missing_tags = []
    ok_cards = []
    recent_note_ids = set()
    total_comparisons = 0

    for note in recent_notes:
        note_id = note["noteId"]
        recent_note_ids.add(note_id)
        fields = note.get("fields", {})
        front = fields.get(fm.front, {}).get("value", "")
        back = fields.get(fm.back, {}).get("value", "")
        note_deck = note.get("_deck", deck)
        existing_tags = set(note.get("tags", []))

        if not front.strip():
            continue

        # Classify
        taxonomy_tags = [t for t in existing_tags if t in valid_tags]
        fp = make_fingerprint(normalize_text(front), normalize_text(back))
        classified = classify_fields(
            front=front,
            back=back,
            source="anki",
            suggested_tags=taxonomy_tags,
            note_type=note.get("modelName", "Basic"),
            taxonomy=taxonomy,
            fingerprint=fp,
            profile_name=profile_name,
            profile=profile_config,
            deck=note_deck,
        )

        # Build query record for similarity search
        query_record = build_record_from_fields(
            candidate_id=f"recent:{note_id}",
            source="audit",
            origin_source="anki",
            front=front,
            back=back,
            scope=classified.scope,
            note_type=classified.note_type,
            deck=note_deck,
            anki_note_id=note_id,
        )

        # Find similar in pool (excluding self)
        pool_without_self = [
            r for r in pool if r.anki_note_id != note_id
        ]
        matches = find_similar_records(
            query_record,
            pool_without_self,
            similar_threshold=settings.acm.similar_lookup_threshold,
        )
        total_comparisons += len(pool_without_self)

        strong_matches = [m for m in matches if m.score >= settings.acm.cluster_threshold]

        # Check missing tags
        missing = _missing_categories(
            categories=required_categories,
            existing_tags=existing_tags,
            classified=classified,
        )

        card_info = {
            "note_id": note_id,
            "front": front[:120],
            "deck": note_deck,
        }

        has_issues = False

        if strong_matches:
            has_issues = True
            duplicates.append({
                **card_info,
                "matches": [
                    {
                        "note_id": m.record.anki_note_id,
                        "front": m.record.front[:120],
                        "score": m.score,
                        "reasons": list(m.reason_codes),
                    }
                    for m in strong_matches[:3]
                ],
            })

        if missing:
            has_issues = True
            new_tags = [t for t in classified.tags_resolved if t not in existing_tags]
            missing_tags.append({
                **card_info,
                "missing_categories": missing,
                "auto_tags": new_tags,
                "scope": classified.scope.summary(),
            })

        if not has_issues:
            ok_cards.append(card_info)

    client.close()

    return json.dumps({
        "recent_count": len(recent_notes),
        "days": days,
        "deck": deck,
        "anki_available": True,
        "embeddings_used": _embeddings_used(settings),
        "duplicates": duplicates,
        "missing_tags": missing_tags,
        "ok": ok_cards,
        "summary": {
            "total": len(recent_notes),
            "duplicates": len(duplicates),
            "missing_tags": len(missing_tags),
            "ok": len(ok_cards),
        },
    })


def main():
    mcp.run()


if __name__ == "__main__":
    main()
