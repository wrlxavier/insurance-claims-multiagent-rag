"""The chunk corpus artifact -- ``build/`` (gitignored) -- [M3-01].

Mirrors [infrastructure.parsing.corpus_artifact]: the combined output of
``scripts/build_chunks.py`` over the whole corpus, kept separate from
``build/manifest.json`` (which [infrastructure.parsing.corpus_artifact.
BuildManifest]'s own docstring scopes to "one ``make parse`` run") since
chunking is a distinct downstream stage, run via its own
``scripts/build_chunks.py`` / ``make build-chunks``.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel

from infrastructure.rag.chunk_schema import ChunkRecord

BUILD_DIR = Path("build")
CHUNKS_PARQUET_PATH = BUILD_DIR / "chunks.parquet"
CHUNKS_JSONL_PATH = BUILD_DIR / "chunks.jsonl"
CHUNKS_MANIFEST_PATH = BUILD_DIR / "chunks_manifest.json"


class ChunksBuildManifest(BaseModel):
    """Reproducibility record for one ``make build-chunks`` run."""

    schema_version: str
    chunking_version: str
    clause_segmentation_version: str
    built_at_utc: datetime
    chunk_counts_by_document: dict[str, int]
    total_chunk_count: int


def utc_now() -> datetime:
    """Return the current time in UTC, for [ChunksBuildManifest.built_at_utc]."""
    return datetime.now(UTC)


def write_chunks_parquet(records: list[ChunkRecord], path: Path) -> None:
    """Write the flattened chunk corpus to Parquet, one row per chunk."""
    rows = [record.model_dump(mode="json") for record in records]
    table = pa.Table.from_pylist(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def read_chunks_parquet(path: Path) -> list[ChunkRecord]:
    """Read the flattened chunk corpus back from Parquet."""
    rows = pq.read_table(path).to_pylist()
    return [ChunkRecord.model_validate(row) for row in rows]


def write_chunks_jsonl(records: list[ChunkRecord], path: Path) -> None:
    """Write the flattened chunk corpus to JSONL, one row per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.model_dump_json())
            handle.write("\n")


def read_chunks_jsonl(path: Path) -> list[ChunkRecord]:
    """Read the flattened chunk corpus back from JSONL."""
    with path.open("r", encoding="utf-8") as handle:
        return [
            ChunkRecord.model_validate(json.loads(line))
            for line in handle
            if line.strip()
        ]


def write_chunks_manifest(manifest: ChunksBuildManifest, path: Path) -> None:
    """Write the chunks build manifest as pretty-printed JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
