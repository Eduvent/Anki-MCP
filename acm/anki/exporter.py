from __future__ import annotations

import csv
import sys
from io import StringIO
from pathlib import Path

from acm.models import ClassifiedCard


def export_tsv(cards: list[ClassifiedCard], path: Path | None = None) -> None:
    """Exporta tarjetas a TSV para importación manual en Anki.

    Si path es None, escribe a stdout.
    Formato: Front\tBack\tTags
    """
    output = open(path, "w", newline="", encoding="utf-8") if path else sys.stdout

    try:
        writer = csv.writer(output, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        for card in cards:
            tags_str = " ".join(card.tags_resolved)
            writer.writerow([card.front, card.back, tags_str])
    finally:
        if path:
            output.close()


def export_rows_tsv(rows, path: Path) -> int:
    """E4-5: exporta filas de processed_cards a TSV (fallback si Anki está cerrado).

    Formato: Front\tBack\tTags. Devuelve cuántas filas escribió.
    """
    count = 0
    with open(path, "w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        for row in rows:
            writer.writerow([row["front_original"], row["back_original"], row["tags_resolved"] or ""])
            count += 1
    return count
