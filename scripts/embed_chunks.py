#!/usr/bin/env python3
"""Embed the chunk corpus and record the cold-run embedding cost -- [M3-02].

Composes the real embedder with the cache and runs the resumable pipeline:
``CachingEmbedder(SentenceTransformerEmbedder(...))`` ->
[infrastructure.rag.embedding_pipeline.embed_missing_chunks]. The cursor is
``WHERE embedding IS NULL``, so a re-run only embeds what is left.

Writes a regenerable cost report to ``eval/runs/embedding_cost_report.{json,md}``
(gitignored, like the ANN benchmark). The committed headline numbers live in
``docs/EMBEDDINGS.md`` and ``README.md``. "Cold run" = the
``data/cache/embeddings/`` cache was empty, so every chunk was a real forward
pass -- that is the number that answers "what does it cost to reproduce this".

``sentence-transformers`` is the optional ``embed`` dependency group:
``make embed-chunks`` runs this via ``uv run --group embed``, and it depends on
``make load-chunks``. ``--dry-run`` (token total + ``$0.00``, no model, no DB)
and the dollar cost need neither the group nor Postgres:

    PYTHONPATH=app/src uv run python scripts/embed_chunks.py --dry-run

The model runs locally, so the dollar cost is $0.00 and no price constant is
introduced (per [M1-09]'s stale-pricing lesson there is nothing to date-stamp
beyond that sentence). The reproducible cost is machine time.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import platform
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session
from tokenizers import Tokenizer

from infrastructure.config.settings import get_embedding_settings
from infrastructure.database import (
    assert_chunk_table_ready,
    create_engine_from_settings,
    create_session_factory,
)
from infrastructure.database.chunk_repository import fetch_chunks_missing_embedding
from infrastructure.database.models import ChunkRow
from infrastructure.rag.chunk_artifact import CHUNKS_JSONL_PATH, read_chunks_jsonl
from infrastructure.rag.chunk_schema import ChunkRecord
from infrastructure.rag.embedder import Embedder
from infrastructure.rag.embedding_cache import CachingEmbedder
from infrastructure.rag.embedding_config import (
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL_ID,
    EMBEDDING_MODEL_REVISION,
    NORMALIZE_EMBEDDINGS,
    format_passage,
)
from infrastructure.rag.embedding_pipeline import EmbeddingRun, embed_missing_chunks

SCHEMA_VERSION = "v1"
OUTPUT_DIR = Path("eval/runs")
JSON_PATH = OUTPUT_DIR / "embedding_cost_report.json"
MD_PATH = OUTPUT_DIR / "embedding_cost_report.md"


class _Encoding(Protocol):
    @property
    def ids(self) -> list[int]: ...


class _Tokenizer(Protocol):
    def encode(self, sequence: str) -> _Encoding: ...


class EmbeddingCostReport(BaseModel):
    """The cost of embedding the chunk corpus once, cache-cold."""

    schema_version: str
    run_at_utc: datetime
    corpus_path: str
    chunk_count: int
    chunks_embedded: int
    chunks_already_present: int
    batches: int
    cache_hits: int
    cache_misses: int
    total_passage_tokens: int
    max_passage_tokens: int
    wall_clock_seconds: float
    tokens_per_second: float
    dollar_cost_usd: float
    cost_note: str
    model_id: str
    model_revision: str
    embedding_dimensions: int
    normalize_embeddings: bool
    batch_size: int
    device: str
    platform: str
    python_version: str
    processor: str
    sentence_transformers_version: str
    torch_version: str
    pgvector_version: str
    postgres_version: str


def load_records(path: Path) -> list[ChunkRecord]:
    """Load the chunk corpus, failing loudly if it has not been built."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Run `make build-chunks` (full rebuild) or "
            "`make fetch-corpus-artifacts` (pre-built corpus) first."
        )
    return read_chunks_jsonl(path)


def passage_token_counts(
    records: list[ChunkRecord], tokenizer: _Tokenizer
) -> list[int]:
    """Token count per chunk, tokenising exactly what the index side embeds.

    Mirrors ``scripts/check_embedding_input_length.token_counts`` -- both run
    ``format_passage`` and count special tokens, so the totals reconcile.
    """
    return [
        len(tokenizer.encode(format_passage(record.text)).ids) for record in records
    ]


def build_cost_note(
    *, run_date: str, chunks_embedded: int, wall_clock_seconds: float, processor: str
) -> str:
    """The dated, prose $0.00 statement copied into docs/EMBEDDINGS.md."""
    zero = (
        f"$0.00 as of {run_date}: {EMBEDDING_MODEL_ID} runs locally via "
        "sentence-transformers -- no API, no per-token charge, and no price "
        "constant is introduced."
    )
    if chunks_embedded == 0:
        return (
            f"{zero} Nothing was embedded this run; clear "
            "`data/cache/embeddings/` and re-run for a cold machine-time figure."
        )
    minutes = wall_clock_seconds / 60
    return (
        f"{zero} The reproducible cost is machine time: ~{minutes:.1f} min for a "
        f"cold pass (empty embedding cache) over {chunks_embedded} chunks on "
        f"{processor}."
    )


def dry_run_summary(records: list[ChunkRecord], tokenizer: _Tokenizer) -> str:
    """Token total + $0.00, without touching the model or the database."""
    counts = passage_token_counts(records, tokenizer)
    return (
        f"embed-chunks --dry-run: {len(records)} chunks, {sum(counts)} passage "
        f"tokens (max {max(counts)}), model {EMBEDDING_MODEL_ID}. Local model, so "
        "$0.00; a real `make embed-chunks` measures the cold-run machine time."
    )


def run_embed_chunks(
    session: Session, embedder: Embedder, *, batch_size: int
) -> tuple[EmbeddingRun, float]:
    """Run the pipeline; return its result and the wall-clock seconds it took."""
    start = time.perf_counter()
    run = embed_missing_chunks(session, embedder, batch_size=batch_size)
    return run, time.perf_counter() - start


def server_versions(session: Session) -> dict[str, str]:
    """The pgvector extension version and the Postgres server version string."""
    pgvector = session.execute(
        text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
    ).scalar_one()
    server = session.execute(text("SELECT version()")).scalar_one()
    return {
        "pgvector_version": f"pgvector {pgvector}",
        "postgres_version": str(server),
    }


def _library_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def _processor_name() -> str:
    """A human CPU name where the OS exposes one, else ``platform`` best-effort."""
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def build_report(
    *,
    run: EmbeddingRun,
    wall_clock_seconds: float,
    token_counts: list[int],
    chunk_count: int,
    cache_hits: int,
    cache_misses: int,
    batch_size: int,
    device: str,
    versions: dict[str, str],
) -> EmbeddingCostReport:
    """Assemble the cost report from a completed run. No IO."""
    total_tokens = sum(token_counts)
    run_at = datetime.now(UTC)
    throughput = total_tokens / wall_clock_seconds if wall_clock_seconds > 0 else 0.0
    return EmbeddingCostReport(
        schema_version=SCHEMA_VERSION,
        run_at_utc=run_at,
        corpus_path=str(CHUNKS_JSONL_PATH),
        chunk_count=chunk_count,
        chunks_embedded=run.embedded,
        chunks_already_present=run.already_present,
        batches=run.batches,
        cache_hits=cache_hits,
        cache_misses=cache_misses,
        total_passage_tokens=total_tokens,
        max_passage_tokens=max(token_counts) if token_counts else 0,
        wall_clock_seconds=round(wall_clock_seconds, 2),
        tokens_per_second=round(throughput, 1),
        dollar_cost_usd=0.0,
        cost_note=build_cost_note(
            run_date=run_at.date().isoformat(),
            chunks_embedded=run.embedded,
            wall_clock_seconds=wall_clock_seconds,
            processor=_processor_name(),
        ),
        model_id=EMBEDDING_MODEL_ID,
        model_revision=EMBEDDING_MODEL_REVISION,
        embedding_dimensions=EMBEDDING_DIMENSIONS,
        normalize_embeddings=NORMALIZE_EMBEDDINGS,
        batch_size=batch_size,
        device=device,
        platform=platform.platform(),
        python_version=platform.python_version(),
        processor=_processor_name(),
        sentence_transformers_version=_library_version("sentence-transformers"),
        torch_version=_library_version("torch"),
        **versions,
    )


def render_markdown_report(report: EmbeddingCostReport) -> str:
    """Render the cost report as Markdown for stdout and the PR."""
    wall = report.wall_clock_seconds
    lines = [
        f"# Corpus embedding cost -- {report.model_id}",
        "",
        "Generated by `scripts/embed_chunks.py` (`make embed-chunks`). "
        "Regenerable; the committed headline numbers live in "
        "`docs/EMBEDDINGS.md` and `README.md`.",
        "",
        "**Cold run** = `data/cache/embeddings/` was empty, so every chunk was a "
        "real model forward pass -- the cost of reproducing the index from "
        "scratch. A warm re-run (cache present, or no NULL vectors left) does "
        "zero inference.",
        "",
        "| measurement | value |",
        "| --- | --- |",
        f"| chunks embedded this run | {report.chunks_embedded} |",
        f"| chunks already embedded (skipped) | {report.chunks_already_present} |",
        f"| batches | {report.batches} |",
        f"| cache hits / misses | {report.cache_hits} / {report.cache_misses} |",
        f"| total passage tokens (corpus) | {report.total_passage_tokens} |",
        f"| max passage tokens (one chunk) | {report.max_passage_tokens} |",
        f"| wall-clock | {wall:.1f} s (~{wall / 60:.1f} min) |",
        f"| throughput | {report.tokens_per_second:.0f} passage tokens/s |",
        f"| **dollar cost** | **${report.dollar_cost_usd:.2f}** (local model) |",
        "",
        report.cost_note,
        "",
        "## Run configuration",
        "",
        f"- model: `{report.model_id}` @ `{report.model_revision}`, "
        f"{report.embedding_dimensions}-dim, normalize={report.normalize_embeddings}",
        f"- batch size: {report.batch_size} (`EMBEDDING_BATCH_SIZE`)",
        f"- device: {report.device}",
        f"- corpus: `{report.corpus_path}`, {report.chunk_count} chunks",
        f"- sentence-transformers {report.sentence_transformers_version}, "
        f"torch {report.torch_version}",
        f"- {report.pgvector_version}; {report.postgres_version}",
        f"- platform: {report.platform}; python {report.python_version}; "
        f"{report.processor}",
        f"- run at (UTC): {report.run_at_utc.isoformat()}",
        "",
    ]
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the corpus token total and $0.00, without the model or a DB",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="force a sentence-transformers device (e.g. 'cpu'); default auto",
    )
    return parser.parse_args()


def main() -> None:
    """Embed the pending chunks and write the cost report."""
    args = _parse_args()
    records = load_records(CHUNKS_JSONL_PATH)
    tokenizer = Tokenizer.from_pretrained(
        EMBEDDING_MODEL_ID, revision=EMBEDDING_MODEL_REVISION
    )

    if args.dry_run:
        print(dry_run_summary(records, tokenizer))
        return

    counts = passage_token_counts(records, tokenizer)
    batch_size = get_embedding_settings().embedding_batch_size

    engine = create_engine_from_settings()
    session = create_session_factory(engine=engine)()
    try:
        assert_chunk_table_ready(session)
        total_chunks: int = session.execute(
            select(func.count()).select_from(ChunkRow)
        ).scalar_one()
        if total_chunks == 0:
            raise RuntimeError("chunk table is empty -- run `make load-chunks` first.")

        if not fetch_chunks_missing_embedding(session):
            print(f"all {total_chunks} chunks already embedded -- nothing to do.")
            print("clear data/cache/embeddings/ and re-run for a cold cost report.")
            return

        versions = server_versions(session)

        # Imported here so `--dry-run` and every code path above stay importable
        # without the optional `embed` group.
        from infrastructure.rag.sentence_transformer_embedder import (
            SentenceTransformerEmbedder,
        )

        inner = SentenceTransformerEmbedder(batch_size=batch_size, device=args.device)
        embedder = CachingEmbedder(inner)
        run, wall = run_embed_chunks(session, embedder, batch_size=batch_size)
        report = build_report(
            run=run,
            wall_clock_seconds=wall,
            token_counts=counts,
            chunk_count=len(records),
            cache_hits=embedder.hits,
            cache_misses=embedder.misses,
            batch_size=batch_size,
            device=inner.device,
            versions=versions,
        )
    finally:
        session.close()
        engine.dispose()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    MD_PATH.write_text(render_markdown_report(report), encoding="utf-8")
    print(
        f"embedded {report.chunks_embedded} chunks in {report.wall_clock_seconds:.1f}s "
        f"on {report.device}; ${report.dollar_cost_usd:.2f}. "
        f"Wrote {JSON_PATH} and {MD_PATH}"
    )


if __name__ == "__main__":
    main()
