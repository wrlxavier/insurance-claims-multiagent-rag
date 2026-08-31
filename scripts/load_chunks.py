#!/usr/bin/env python3
"""Upsert the built chunk corpus into the ``chunk`` table -- [M3-02].

Reads ``build/chunks.jsonl`` ([M3-01]'s output) and writes every row into
Postgres via
[infrastructure.database.chunk_repository.upsert_chunks]. Metadata only -- it
never touches ``chunk.embedding`` (``make embed-chunks`` owns that column).

Idempotent: ``chunk_id`` is deterministic upstream, so ``INSERT ... ON CONFLICT
(chunk_id) DO UPDATE`` means a re-run over the same corpus neither duplicates
rows nor needs a wipe, and it leaves vectors already computed in place.

Run via ``make load-chunks`` after ``make build-chunks`` (or
``make fetch-corpus-artifacts``) and ``make migrate``. This is the first
``make embed-chunks`` step and the "embed" stage [M3-08]'s ``make build-index``
composes.
"""

from __future__ import annotations

from pathlib import Path

from infrastructure.database import (
    assert_chunk_table_ready,
    create_engine_from_settings,
    create_session_factory,
    upsert_chunks,
)
from infrastructure.rag.chunk_artifact import CHUNKS_JSONL_PATH, read_chunks_jsonl
from infrastructure.rag.chunk_schema import ChunkRecord


def load_records(path: Path) -> list[ChunkRecord]:
    """Load the chunk corpus, failing loudly if it has not been built."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Run `make build-chunks` (full rebuild) or "
            "`make fetch-corpus-artifacts` (pre-built corpus) first."
        )
    return read_chunks_jsonl(path)


def main() -> None:
    """Upsert every built chunk into the ``chunk`` table."""
    records = load_records(CHUNKS_JSONL_PATH)
    engine = create_engine_from_settings()
    session = create_session_factory(engine=engine)()
    try:
        assert_chunk_table_ready(session)
        written = upsert_chunks(session, records)
        session.commit()
    finally:
        session.close()
        engine.dispose()
    print(f"upserted {written} chunks into chunk (from {CHUNKS_JSONL_PATH})")


if __name__ == "__main__":
    main()
