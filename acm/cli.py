from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import typer
import yaml
from rich import print as rprint
from rich.console import Console
from rich.table import Table

from acm.anki.client import AnkiConnectClient, AnkiConnectError
from acm.anki.exporter import export_tsv
from acm.config import load_profile_taxonomy, load_settings, save_taxonomy
from acm.models import CandidateCard, CardScope
from acm.pipeline.auditor import audit_batch
from acm.pipeline.duplicate_audit import find_duplicate_clusters, find_similar_card
from acm.pipeline.similarity import serialize_cluster, serialize_match, serialize_metrics
from acm.service import (
    resolve_record as _resolve_record,
    scope_from_row as _scope_from_row,
    sync_pending as _sync_pending,
)
from acm.store.registry import Registry

app = typer.Typer(
    name="acm",
    help="Anki Card Manager — ingest, deduplica y sincroniza tarjetas con Anki.",
    no_args_is_help=True,
)
taxonomy_app = typer.Typer(help="Gestiona la taxonomía de tags.")
app.add_typer(taxonomy_app, name="taxonomy")

console = Console()


def _get_registry(settings_path: Path | None = None) -> tuple[Registry, object]:
    settings = load_settings(settings_path)
    registry = Registry(settings.db_path_resolved)
    return registry, settings


def _get_context(
    profile_name: str | None = None,
    settings_path: Path | None = None,
):
    registry, settings = _get_registry(settings_path)
    resolved_profile_name, profile, taxonomy = load_profile_taxonomy(settings, profile_name)
    return registry, settings, resolved_profile_name, profile, taxonomy


def _scope_summary(scope: CardScope) -> str:
    if not scope.facets:
        return "-"
    return ", ".join(f"{key}={value}" for key, value in sorted(scope.summary().items()))


def _action_style(action: str) -> str:
    return {
        "insert": "[green]insert[/green]",
        "possible_duplicate": "[yellow]possible_duplicate[/yellow]",
        "reject": "[red]reject[/red]",
    }.get(action, action)


def _print_json(payload: dict) -> None:
    console.print_json(json.dumps(payload, ensure_ascii=False))


# ---------------------------------------------------------------------------
# acm ingest
# ---------------------------------------------------------------------------

@app.command()
def ingest(
    file: Optional[Path] = typer.Argument(None, help="JSON/YAML con tarjetas candidatas"),
    stdin: bool = typer.Option(False, "--stdin", help="Leer desde stdin"),
    export_tsv_path: Optional[Path] = typer.Option(None, "--export-tsv", help="Exportar aprobadas a TSV"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Mostrar decisiones sin guardar"),
    profile: Optional[str] = typer.Option(None, "--profile", help="Perfil de clasificación a usar"),
    deck: Optional[str] = typer.Option(None, "--deck", help="Deck objetivo para auditar/sincronizar"),
) -> None:
    """Procesa tarjetas candidatas: normaliza, clasifica, deduplica y decide."""
    if stdin:
        raw = sys.stdin.read()
    elif file:
        raw = file.read_text()
    else:
        rprint("[red]Error:[/red] Proporciona un archivo o usa --stdin")
        raise typer.Exit(1)

    # Parsear JSON o YAML
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        try:
            data = yaml.safe_load(raw)
        except yaml.YAMLError as e:
            rprint(f"[red]Error parseando input:[/red] {e}")
            raise typer.Exit(1)

    if not isinstance(data, list):
        rprint("[red]Error:[/red] El input debe ser una lista de tarjetas")
        raise typer.Exit(1)

    cards: list[CandidateCard] = []
    for i, item in enumerate(data):
        try:
            if profile and "profile" not in item:
                item["profile"] = profile
            if deck and "deck" not in item:
                item["deck"] = deck
            cards.append(CandidateCard(**item))
        except Exception as e:
            rprint(f"[yellow]Advertencia:[/yellow] Tarjeta {i} inválida, omitiendo: {e}")

    if not cards:
        rprint("[yellow]Sin tarjetas válidas para procesar.[/yellow]")
        raise typer.Exit(0)

    registry, settings, resolved_profile_name, resolved_profile, taxonomy = _get_context(profile)

    # Intentar conectar con Anki
    anki_client: AnkiConnectClient | None = None
    with AnkiConnectClient(settings.anki.connect_url) as client:
        if client.is_available():
            anki_client = client
            rprint("[dim]AnkiConnect disponible.[/dim]")
        else:
            rprint("[yellow]Anki no disponible. Dedupe solo contra registro local.[/yellow]")

        decisions = audit_batch(
            cards,
            registry,
            taxonomy,
            settings,
            anki_client,
            profile_name=resolved_profile_name,
            profile=resolved_profile,
        )

    # Mostrar resultados
    table = Table(title=f"Resultados — {len(decisions)} tarjeta(s)", show_lines=True)
    table.add_column("#", style="dim", width=4)
    table.add_column("Acción", width=20)
    table.add_column("Front", max_width=45)
    table.add_column("Razón", max_width=50)

    for i, d in enumerate(decisions, 1):
        table.add_row(
            str(i),
            _action_style(d.action),
            d.card.front[:80],
            d.reason,
        )
    console.print(table)

    if not dry_run:
        for d in decisions:
            registry.insert(d)
        rprint(f"[green]Guardadas {len(decisions)} decisiones en el registro.[/green]")

        if export_tsv_path:
            insert_cards = [d.card for d in decisions if d.action == "insert"]
            export_tsv(insert_cards, export_tsv_path)
            rprint(f"[green]TSV exportado a {export_tsv_path}[/green]")

    # Resumen
    by_action: dict[str, int] = {}
    for d in decisions:
        by_action[d.action] = by_action.get(d.action, 0) + 1
    parts = [f"{v} {k}" for k, v in by_action.items()]
    rprint(f"\n[bold]Resumen:[/bold] {' · '.join(parts)}")

    if any(d.card.tags_unresolved for d in decisions):
        rprint("[yellow]Hay tags no resueltos. Usa 'acm taxonomy show' para ver los válidos.[/yellow]")


# ---------------------------------------------------------------------------
# acm review
# ---------------------------------------------------------------------------

@app.command()
def review() -> None:
    """Muestra tarjetas en estado possible_duplicate pendientes de revisión."""
    registry, _ = _get_registry()
    rows = registry.list_pending_review()

    if not rows:
        rprint("[green]Sin tarjetas pendientes de revisión.[/green]")
        return

    table = Table(title=f"Pendientes de revisión — {len(rows)}", show_lines=True)
    table.add_column("ID", style="dim", max_width=10)
    table.add_column("Front", max_width=46)
    table.add_column("Estado", width=12)
    table.add_column("Perfil", width=12)
    table.add_column("Deck", max_width=18)
    table.add_column("Scope", max_width=22)
    table.add_column("Fecha", width=20)

    for row in rows:
        short_id = row["id"][:8]
        scope = _scope_from_row(row)
        estado = row["status"] if "status" in row.keys() else "-"
        table.add_row(
            short_id,
            row["front_normalized"][:80],
            estado,
            row["profile_name"] or "-",
            row["target_deck"] or "-",
            _scope_summary(scope),
            row["created_at"][:19],
        )
    console.print(table)
    rprint("[dim]Usa 'acm approve <id>' o 'acm reject <id>' para decidir.[/dim]")


# ---------------------------------------------------------------------------
# acm approve / reject
# ---------------------------------------------------------------------------

def _cli_resolve(record_id: str, action: str) -> dict:
    registry, _ = _get_registry()
    result = _resolve_record(registry, record_id, action)
    if "error" in result:
        if "matches" in result:
            rprint(f"[yellow]Prefijo ambiguo:[/yellow] {result['matches']}")
        else:
            rprint(f"[red]{result['error']}[/red]")
        raise typer.Exit(1)
    return result


@app.command()
def approve(record_id: str = typer.Argument(..., help="ID (o prefijo) del registro")) -> None:
    """Aprueba una tarjeta en revisión y la marca para inserción en Anki."""
    result = _cli_resolve(record_id, "approve")
    rprint(f"[green]Aprobada:[/green] {result['id'][:8]} — marcada para el próximo sync.")


@app.command()
def reject(record_id: str = typer.Argument(..., help="ID (o prefijo) del registro")) -> None:
    """Descarta definitivamente una tarjeta."""
    result = _cli_resolve(record_id, "reject")
    rprint(f"[red]Rechazada:[/red] {result['id'][:8]}")


# ---------------------------------------------------------------------------
# acm sync
# ---------------------------------------------------------------------------

@app.command()
def sync(
    export_tsv_fallback: Optional[Path] = typer.Option(None, "--export-tsv", help="Exportar a TSV si Anki no disponible"),
) -> None:
    """Empuja todas las tarjetas aprobadas pendientes a Anki (vía capa de servicio)."""
    registry, settings = _get_registry()
    result = _sync_pending(registry, settings, export_tsv_path=export_tsv_fallback)

    if result.get("error"):
        rprint(f"[yellow]{result['error']}[/yellow]")
        if result.get("exported_tsv"):
            rprint(f"[green]Fallback: {result['exported_count']} aprobada(s) exportadas a {result['exported_tsv']}[/green]")
        raise typer.Exit(1)

    if result.get("synced_count", 0) == 0 and not result.get("errors"):
        rprint("[green]Sin tarjetas pendientes de sync.[/green]")
        return

    for item in result["synced"]:
        rprint(f"  [green]✓[/green] {item['id']} → [dim]{item['deck']}[/dim] (note {item['note_id']})")
    for err in result["errors"]:
        rprint(f"  [red]✗[/red] {err['id']}: {err['error']}")

    rprint(f"[green]Sincronizadas {result['synced_count']} tarjeta(s).[/green]")
    if result["errors"]:
        rprint(f"[red]{result['error_count']} error(es) durante el sync.[/red]")


# ---------------------------------------------------------------------------
# acm audit-duplicates
# ---------------------------------------------------------------------------

@app.command("audit-duplicates")
def audit_duplicates(
    deck: str = typer.Option(..., "--deck", help="Deck raíz a auditar"),
    profile: Optional[str] = typer.Option(None, "--profile", help="Perfil de clasificación del deck"),
    include_subdecks: Optional[bool] = typer.Option(
        None,
        "--include-subdecks/--no-include-subdecks",
        help="Incluir subdecks del deck objetivo",
    ),
    output_format: str = typer.Option("table", "--format", help="table | json"),
) -> None:
    """Busca clusters de tarjetas repetidas dentro del deck objetivo y el registro local."""
    registry, settings, resolved_profile_name, resolved_profile, taxonomy = _get_context(profile)
    include_subdecks = settings.acm.audit_include_subdecks if include_subdecks is None else include_subdecks
    if output_format not in {"table", "json"}:
        rprint("[red]Formato inválido.[/red] Usa table o json.")
        raise typer.Exit(1)

    clusters = []
    metrics = None
    with AnkiConnectClient(settings.anki.connect_url) as client:
        anki_available = client.is_available()
        if not anki_available and not registry.list_indexed_notes(deck_name=deck, include_subdecks=include_subdecks):
            rprint("[red]Anki no disponible y no hay índice local para ese deck.[/red]")
            raise typer.Exit(1)

        clusters, metrics = find_duplicate_clusters(
            registry=registry,
            settings=settings,
            taxonomy=taxonomy,
            profile_name=resolved_profile_name,
            profile=resolved_profile,
            deck=deck,
            include_subdecks=include_subdecks,
            include_registry=True,
            anki_client=client if anki_available else None,
            refresh_index=anki_available,
        )

    payload = {
        "deck": deck,
        "include_subdecks": include_subdecks,
        "clusters": [serialize_cluster(cluster) for cluster in clusters],
        "metrics": serialize_metrics(metrics),
    }
    if output_format == "json":
        _print_json(payload)
        return

    if not clusters:
        rprint("[green]Sin clusters de duplicados fuertes.[/green]")
        rprint(
            "[dim]"
            f"Comparaciones: {metrics.comparisons_run} "
            f"(reducción {metrics.comparison_reduction_pct:.2f}% vs N²)"
            "[/dim]"
        )
        return

    table = Table(title=f"Clusters de duplicados — {len(clusters)}", show_lines=True)
    table.add_column("Cluster", style="dim", width=12)
    table.add_column("Miembros", justify="right", width=8)
    table.add_column("Representante", max_width=42)
    table.add_column("Razones", max_width=32)
    table.add_column("Score", justify="right", width=8)

    for cluster in clusters:
        table.add_row(
            cluster.cluster_id[:12],
            str(len(cluster.members)),
            cluster.representative.front[:80],
            ", ".join(cluster.reason_codes),
            f"{cluster.score_floor:.2f}",
        )
    console.print(table)
    rprint(
        "[dim]"
        f"Cards: {metrics.cards_scanned} · "
        f"Candidatos: {metrics.candidates_generated} · "
        f"Comparaciones: {metrics.comparisons_run} · "
        f"Reducción: {metrics.comparison_reduction_pct:.2f}%"
        "[/dim]"
    )


# ---------------------------------------------------------------------------
# acm similar
# ---------------------------------------------------------------------------

@app.command()
def similar(
    front: str = typer.Option(..., "--front", help="Front a evaluar"),
    back: str = typer.Option("", "--back", help="Back opcional para desempate"),
    deck: Optional[str] = typer.Option(None, "--deck", help="Deck donde buscar primero"),
    profile: Optional[str] = typer.Option(None, "--profile", help="Perfil de clasificación a usar"),
    note_type: Optional[str] = typer.Option(None, "--note-type", help="Tipo de nota del query"),
    include_subdecks: Optional[bool] = typer.Option(
        None,
        "--include-subdecks/--no-include-subdecks",
        help="Incluir subdecks cuando se consulta un deck",
    ),
) -> None:
    """Busca tarjetas similares usando el motor local de duplicados."""
    registry, settings, resolved_profile_name, resolved_profile, taxonomy = _get_context(profile)
    include_subdecks = settings.acm.audit_include_subdecks if include_subdecks is None else include_subdecks
    note_type = note_type or settings.anki.default_model or "Basic"

    with AnkiConnectClient(settings.anki.connect_url) as client:
        anki_available = client.is_available()
        if deck and not anki_available and not registry.list_indexed_notes(deck_name=deck, include_subdecks=include_subdecks):
            rprint("[red]Anki no disponible y no hay índice local para ese deck.[/red]")
            raise typer.Exit(1)

        matches = find_similar_card(
            registry=registry,
            settings=settings,
            taxonomy=taxonomy,
            profile_name=resolved_profile_name,
            profile=resolved_profile,
            front=front,
            back=back,
            note_type=note_type,
            deck=deck,
            include_subdecks=include_subdecks,
            anki_client=client if anki_available else None,
        )

    if not matches:
        rprint("[green]Sin coincidencias por encima del umbral de similitud.[/green]")
        return

    table = Table(title=f"Similares — {len(matches)} coincidencia(s)", show_lines=True)
    table.add_column("ID", style="dim", max_width=12)
    table.add_column("Source", width=10)
    table.add_column("Deck", max_width=22)
    table.add_column("Score", justify="right", width=7)
    table.add_column("Razones", max_width=28)
    table.add_column("Front", max_width=42)

    for match in matches:
        payload = serialize_match(match)
        table.add_row(
            payload["id"][:12],
            payload["source"],
            payload["deck"] or "-",
            f"{payload['score']:.2f}",
            ", ".join(payload["reason_codes"]),
            payload["front"][:80],
        )
    console.print(table)


# ---------------------------------------------------------------------------
# acm stats
# ---------------------------------------------------------------------------

@app.command()
def stats() -> None:
    """Muestra un resumen de la colección gestionada."""
    registry, _ = _get_registry()
    data = registry.stats()

    if not data:
        rprint("[dim]Sin datos en el registro todavía.[/dim]")
        return

    table = Table(title="Estadísticas del registro")
    table.add_column("Acción", style="bold")
    table.add_column("Cantidad", justify="right")

    total = 0
    for action in ("insert", "possible_duplicate", "reject"):
        count = data.get(action, 0)
        total += count
        table.add_row(_action_style(action), str(count))
    table.add_row("[bold]Total[/bold]", f"[bold]{total}[/bold]")

    console.print(table)


# ---------------------------------------------------------------------------
# acm taxonomy
# ---------------------------------------------------------------------------

@taxonomy_app.command("show")
def taxonomy_show(
    profile: Optional[str] = typer.Option(None, "--profile", help="Perfil cuya taxonomía quieres ver"),
) -> None:
    """Muestra los tags permitidos en la taxonomía."""
    _, settings, resolved_profile_name, _, taxonomy = _get_context(profile)

    rprint(f"[bold]Perfil:[/bold] {resolved_profile_name}")
    for category in taxonomy.category_names():
        values = taxonomy.values_for(category)
        rprint(f"[bold]{category}:[/bold] {', '.join(values) if values else '(vacío)'}")


@taxonomy_app.command("add")
def taxonomy_add(
    category: str = typer.Argument(..., help="Categoría de la taxonomía"),
    value: str = typer.Argument(..., help="Valor a agregar"),
    profile: Optional[str] = typer.Option(None, "--profile", help="Perfil cuya taxonomía quieres editar"),
) -> None:
    """Agrega un tag a la taxonomía."""
    _, settings, _, resolved_profile, taxonomy = _get_context(profile)
    taxonomy_path = resolved_profile.taxonomy_path_resolved(settings.taxonomy_path_resolved)

    current = taxonomy.values_for(category)
    if value in current:
        rprint(f"[yellow]'{value}' ya existe en '{category}'.[/yellow]")
        return

    taxonomy.append_value(category, value)
    save_taxonomy(taxonomy, taxonomy_path)
    rprint(f"[green]Agregado:[/green] {category}::{value}")
