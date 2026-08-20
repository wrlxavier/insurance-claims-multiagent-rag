"""Manifest reading for the extraction pipeline."""

import csv
from pathlib import Path


def read_manifest(path: Path) -> list[dict[str, str]]:
    """Read manifest.csv rows as plain dicts, one per document."""
    with path.open(newline="", encoding="utf-8") as manifest_file:
        return list(csv.DictReader(manifest_file))
