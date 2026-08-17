"""The final, versioned corpus artifact -- ``build/`` (gitignored).

Unlike the per-document caches in ``data/cache/`` (one Parquet file per
document, keyed by a stage-version hash -- see [infrastructure.parsing.
clause_tree_caching] and friends), this is the single combined output of
the whole pipeline: everything downstream of parsing (retrieval indexing,
evaluation, a future golden-set validation step) reads from here, never
from the per-stage caches directly. Rebuilt end to end by ``make parse`` ->
``scripts/build_corpus.py``, never hand-edited.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel

from infrastructure.parsing.clause_schema import ParsedClauseRecord

BUILD_DIR = Path("build")
PARQUET_PATH = BUILD_DIR / "parsed_clauses.parquet"
JSONL_PATH = BUILD_DIR / "parsed_clauses.jsonl"
BUILD_MANIFEST_PATH = BUILD_DIR / "manifest.json"


class BuildManifest(BaseModel):
    """Reproducibility record for one ``make parse`` run.

    ``llm_classification_enabled`` is always ``False`` today -- see
    [infrastructure.parsing.null_classifier] for why ``scripts/
    build_corpus.py`` runs a deterministic stub instead of a real LLM.
    """

    schema_version: str
    clause_segmentation_version: str
    boilerplate_removal_version: str
    llm_classification_enabled: bool
    built_at_utc: datetime
    clause_counts_by_document: dict[str, int]
    total_clause_count: int


def utc_now() -> datetime:
    """Return the current time in UTC, for [BuildManifest.built_at_utc]."""
    return datetime.now(UTC)


def write_parsed_clauses_parquet(records: list[ParsedClauseRecord], path: Path) -> None:
    """Write the flattened corpus to Parquet, one row per clause."""
    rows = [record.model_dump(mode="json") for record in records]
    table = pa.Table.from_pylist(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def read_parsed_clauses_parquet(path: Path) -> list[ParsedClauseRecord]:
    """Read the flattened corpus back from Parquet."""
    rows = pq.read_table(path).to_pylist()
    return [ParsedClauseRecord.model_validate(row) for row in rows]


def write_parsed_clauses_jsonl(records: list[ParsedClauseRecord], path: Path) -> None:
    """Write the flattened corpus to JSONL, one row per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.model_dump_json())
            handle.write("\n")


def read_parsed_clauses_jsonl(path: Path) -> list[ParsedClauseRecord]:
    """Read the flattened corpus back from JSONL."""
    with path.open("r", encoding="utf-8") as handle:
        return [
            ParsedClauseRecord.model_validate(json.loads(line))
            for line in handle
            if line.strip()
        ]


def write_build_manifest(manifest: BuildManifest, path: Path) -> None:
    """Write the build manifest as pretty-printed JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
