#!/usr/bin/env python3
"""Benchmark the HNSW ANN index over ``chunk.embedding`` -- [M3-02].

Resolves three [M3-02] DoD items:

* **build time + index size** -- times ``CREATE INDEX`` and reads
  ``pg_relation_size``;
* **does ANN earn its place at ~4,540 chunks** -- measures exact ``<=>`` scan
  latency against HNSW latency, full-corpus and over a single
  metadata-filtered partition ([M3-04]'s default retrieval path);
* **filtered search returning fewer than k** -- counts the rows an exact search,
  an HNSW search (no iterative scan) and an HNSW search with
  ``hnsw.iterative_scan = strict_order`` return for the default filter and for a
  restrictive stacked filter.

**Synthetic vectors.** No real embedder exists yet (a separate, still-deferred
[M3-02] slice), so ``chunk.embedding`` is filled with deterministic
pseudo-random unit vectors. Build time, index size and *latency* are structural
-- they depend on row count, dimensionality, ``halfvec`` storage and the index
parameters, not on what the vectors mean -- so they transfer to real embeddings.
ANN *recall* vs. exact does **not** transfer (random vectors have no cluster
structure); it is reported only as a sanity check, and the real number is
[M3-08]'s to measure on real embeddings.

Runs against ``TEST_DATABASE_URL`` only, inside one transaction that is always
rolled back -- nothing persists, and any existing ``chunk`` rows are restored.
Requires the ``chunk`` table (``DATABASE_URL=$TEST_DATABASE_URL make migrate``)
and the built chunk corpus (``make build-chunks`` or
``make fetch-corpus-artifacts``). Run via ``make benchmark-ann-index``. Writes
``eval/runs/ann_index_benchmark.{md,json}`` (gitignored); the committed numbers
live in ``docs/EMBEDDINGS.md``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import statistics
import time
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple

from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.orm import Session

from infrastructure.database import (
    create_engine_from_database_url,
    create_session_factory,
)
from infrastructure.database.chunk_repository import (
    upsert_chunks,
    write_chunk_embeddings,
)
from infrastructure.rag.ann_index import (
    HNSW_EF_CONSTRUCTION,
    HNSW_EF_SEARCH,
    HNSW_M,
    INDEX_NAME,
    apply_ann_search_gucs,
    create_hnsw_index,
)
from infrastructure.rag.chunk_artifact import CHUNKS_JSONL_PATH, read_chunks_jsonl
from infrastructure.rag.chunk_schema import ChunkRecord
from infrastructure.rag.embedding_config import EMBEDDING_DIMENSIONS

SCHEMA_VERSION = "v1"
OUTPUT_DIR = Path("eval/runs")
JSON_PATH = OUTPUT_DIR / "ann_index_benchmark.json"
MD_PATH = OUTPUT_DIR / "ann_index_benchmark.md"

VECTOR_SEED = 20260828
QUERY_SEED = 920260828
QUERY_COUNT = 50
WARMUP_ITERATIONS = 3
MEASURE_ITERATIONS = 10
K = 10

_CAVEAT = (
    "**Synthetic vectors.** `chunk.embedding` is filled with deterministic "
    "pseudo-random unit vectors -- no real embedder exists yet. Build time, "
    "index size and latency are structural and transfer to real embeddings; "
    "**ANN recall vs. exact does not** (random vectors have no cluster "
    "structure) and is [M3-08]'s to measure on real embeddings."
)

_ORDER_BY_DISTANCE = "ORDER BY embedding <=> CAST(:q AS halfvec) LIMIT :k"
_DEFAULT_FILTER = "susep_process = :susep_process AND cnpj = :cnpj"


# --------------------------------------------------------------------------- #
# Pure functions                                                             #
# --------------------------------------------------------------------------- #


def pseudo_random_unit_vector(
    key: str, *, dim: int = EMBEDDING_DIMENSIONS
) -> list[float]:
    """A deterministic, L2-normalised pseudo-random vector of the pinned width.

    Seeded from ``sha256(key)`` and drawn from a Gaussian, so components vary in
    sign and the directions spread over the sphere -- unlike the sha256-byte
    ``FakeEmbedder`` in the test suite, whose all-positive-orthant vectors are
    near-parallel and useless for latency or recall realism.
    """
    seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")
    rng = random.Random(seed)
    raw = [rng.gauss(0.0, 1.0) for _ in range(dim)]
    norm = math.sqrt(sum(value * value for value in raw))
    return [value / norm for value in raw]


def generate_query_vectors(count: int, seed: int) -> list[list[float]]:
    """``count`` deterministic query vectors, independent of the corpus keyspace."""
    return [pseudo_random_unit_vector(f"query:{seed}:{i}") for i in range(count)]


def vector_literal(values: Sequence[float]) -> str:
    """Render a vector in pgvector's ``[v1,v2,...]`` text form."""
    return "[" + ",".join(repr(float(value)) for value in values) + "]"


def recall_at_k(approx_ids: Sequence[str], exact_ids: Sequence[str]) -> float:
    """Fraction of the exact top-k also returned by the approximate search."""
    if not exact_ids:
        return 1.0
    return len(set(approx_ids) & set(exact_ids)) / len(exact_ids)


def _percentile(sorted_values: list[float], fraction: float) -> float:
    """Nearest-rank percentile, same formula as the other measurement scripts."""
    index = min(int(len(sorted_values) * fraction), len(sorted_values) - 1)
    return sorted_values[index]


def summarise_latency(samples_ms: list[float]) -> dict[str, float]:
    """Summarise per-call latency samples (milliseconds)."""
    if not samples_ms:
        return {"n": 0, "p50": 0.0, "p95": 0.0, "mean": 0.0}
    ordered = sorted(samples_ms)
    return {
        "n": len(ordered),
        "p50": round(_percentile(ordered, 0.50), 3),
        "p95": round(_percentile(ordered, 0.95), 3),
        "mean": round(statistics.fmean(ordered), 3),
    }


class Partition(NamedTuple):
    """One ``(susep_process, cnpj)`` retrieval partition and its chunk count."""

    susep_process: str
    cnpj: str
    size: int

    @property
    def params(self) -> dict[str, str]:
        """The bind parameters for the default metadata filter."""
        return {"susep_process": self.susep_process, "cnpj": self.cnpj}


def choose_partitions(records: Sequence[ChunkRecord]) -> tuple[Partition, Partition]:
    """The smallest and the median-sized ``(susep_process, cnpj)`` partition."""
    counts = Counter((record.susep_process, record.cnpj) for record in records)
    ordered = sorted(counts.items(), key=lambda item: item[1])
    (small_process, small_cnpj), small_size = ordered[0]
    (median_process, median_cnpj), median_size = ordered[len(ordered) // 2]
    return (
        Partition(small_process, small_cnpj, small_size),
        Partition(median_process, median_cnpj, median_size),
    )


def choose_stacked_clause_type(
    records: Sequence[ChunkRecord], partition: Partition
) -> tuple[str, int] | None:
    """A ``clause_type`` within ``partition`` matching between 1 and k-1 chunks.

    This is the "stacked filter genuinely holds fewer than k rows" case -- a
    real insufficient-context signal, not an index artifact. ``None`` if every
    clause type in the partition has 0 or >= k chunks.
    """
    counts = Counter(
        record.clause_type.value
        for record in records
        if record.susep_process == partition.susep_process
        and record.cnpj == partition.cnpj
    )
    candidates = [(name, n) for name, n in counts.items() if 1 <= n < K]
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[1])


# --------------------------------------------------------------------------- #
# Database access                                                            #
# --------------------------------------------------------------------------- #


def resolve_test_database_url() -> str:
    """``TEST_DATABASE_URL`` from the environment, else from ``.env``.

    Mirrors ``scripts/run_integration_tests.sh``: the shell does not read
    ``.env`` the way pydantic-settings does, so grep the one key rather than
    sourcing the file.
    """
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        env_file = Path(".env")
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                if line.startswith("TEST_DATABASE_URL="):
                    url = line.split("=", 1)[1].strip()
                    break
    if not url:
        raise RuntimeError(
            "TEST_DATABASE_URL must be set (in the environment or .env). This "
            "benchmark truncates and rewrites the `chunk` table, so it runs "
            "against the test database only."
        )
    return url


def assert_target_is_test_db(url: str, *, allow_nontest: bool) -> None:
    """Refuse a database whose name has no ``test`` in it, unless overridden."""
    name = url.rsplit("/", 1)[-1].split("?", 1)[0]
    if "test" not in name and not allow_nontest:
        raise RuntimeError(
            f"refusing to run against database {name!r}: the name has no "
            "'test' in it and this benchmark truncates the `chunk` table. Pass "
            "--allow-nontest-db to override."
        )


def assert_chunk_table_ready(session: Session) -> None:
    """Fail loudly with the fix command if the ``chunk`` table is not migrated."""
    ready = session.execute(
        text(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_name = 'chunk' AND column_name = 'embedding'"
        )
    ).first()
    if ready is None:
        raise RuntimeError(
            "`chunk` table (with the `embedding` column) not found. Run: "
            "DATABASE_URL=$TEST_DATABASE_URL make migrate"
        )


def server_versions(session: Session) -> dict[str, str]:
    """The pgvector extension version and the Postgres server version string."""
    pgvector = session.execute(
        text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
    ).scalar_one()
    server = session.execute(text("SELECT version()")).scalar_one()
    return {"pgvector_version": str(pgvector), "postgres_version": str(server)}


def load_and_embed(
    session: Session, records: Sequence[ChunkRecord], *, seed: int
) -> None:
    """Replace the ``chunk`` table with ``records`` and synthetic embeddings."""
    session.execute(text("TRUNCATE chunk"))
    upsert_chunks(session, list(records))
    vectors = {
        record.chunk_id: pseudo_random_unit_vector(f"{seed}:{record.chunk_id}")
        for record in records
    }
    write_chunk_embeddings(session, vectors)
    session.flush()
    session.execute(text("ANALYZE chunk"))


def build_and_measure_index(session: Session) -> dict[str, Any]:
    """Time ``CREATE INDEX`` and record the index and table sizes."""
    start = time.perf_counter()
    create_hnsw_index(session)
    session.flush()
    build_seconds = time.perf_counter() - start
    index_bytes = session.execute(
        text("SELECT pg_relation_size(CAST(:name AS regclass))"),
        {"name": INDEX_NAME},
    ).scalar_one()
    index_pretty = session.execute(
        text("SELECT pg_size_pretty(pg_relation_size(CAST(:name AS regclass)))"),
        {"name": INDEX_NAME},
    ).scalar_one()
    table_bytes = session.execute(text("SELECT pg_relation_size('chunk')")).scalar_one()
    return {
        "build_seconds": round(build_seconds, 3),
        "index_bytes": int(index_bytes),
        "index_size_pretty": str(index_pretty),
        "table_bytes": int(table_bytes),
        "index_to_table_ratio": round(int(index_bytes) / int(table_bytes), 3),
    }


def _scan_summary(node: dict[str, Any]) -> str:
    """The first scan node in an EXPLAIN plan tree, as ``Type [using index]``."""
    stack: list[dict[str, Any]] = [node]
    while stack:
        current = stack.pop()
        node_type = str(current.get("Node Type", ""))
        if "Scan" in node_type:
            index_name = current.get("Index Name")
            return f"{node_type} using {index_name}" if index_name else node_type
        children = current.get("Plans", [])
        if isinstance(children, list):
            stack.extend(child for child in children if isinstance(child, dict))
    return "unknown"


def explain_scan(session: Session, sql: str, params: dict[str, Any]) -> str:
    """Run ``EXPLAIN (FORMAT JSON)`` and summarise the access path."""
    plan = session.execute(text(f"EXPLAIN (FORMAT JSON) {sql}"), params).scalar_one()
    if isinstance(plan, str):
        plan = json.loads(plan)
    return _scan_summary(plan[0]["Plan"])


def time_query(
    session: Session,
    sql: str,
    params: dict[str, Any],
    *,
    warmup: int = WARMUP_ITERATIONS,
    iterations: int = MEASURE_ITERATIONS,
) -> list[float]:
    """Per-call wall-clock latency (ms) for ``sql``, after ``warmup`` discards."""
    for _ in range(warmup):
        session.execute(text(sql), params).all()
    samples: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        session.execute(text(sql), params).all()
        samples.append((time.perf_counter() - start) * 1000.0)
    return samples


def _topk_ids(session: Session, sql: str, params: dict[str, Any]) -> list[str]:
    return list(session.execute(text(sql), params).scalars())


def measure_latency(
    session: Session,
    *,
    where: str,
    base_params: dict[str, Any],
    query_vectors: Sequence[Sequence[float]],
) -> dict[str, Any]:
    """Aggregate latency over every query vector for one filter."""
    clause = f"WHERE {where} " if where else ""
    sql = f"SELECT chunk_id FROM chunk {clause}{_ORDER_BY_DISTANCE}"
    samples: list[float] = []
    for vector in query_vectors:
        params = {**base_params, "q": vector_literal(vector), "k": K}
        samples.extend(time_query(session, sql, params))
    example = {**base_params, "q": vector_literal(query_vectors[0]), "k": K}
    return {
        "latency_ms": summarise_latency(samples),
        "plan": explain_scan(session, sql, example),
    }


def measure_recall(session: Session, query_vectors: Sequence[Sequence[float]]) -> float:
    """Mean plain-HNSW recall@k against the exact top-k, over every query vector."""
    sql = f"SELECT chunk_id FROM chunk {_ORDER_BY_DISTANCE}"
    apply_ann_search_gucs(session, iterative_scan="off")  # plain ef_search recall
    recalls: list[float] = []
    for vector in query_vectors:
        params = {"q": vector_literal(vector), "k": K}
        session.execute(text("SET LOCAL enable_indexscan = off"))
        session.execute(text("SET LOCAL enable_indexonlyscan = off"))
        exact_ids = _topk_ids(session, sql, params)
        session.execute(text("SET LOCAL enable_indexscan = on"))
        session.execute(text("SET LOCAL enable_indexonlyscan = on"))
        approx_ids = _topk_ids(session, sql, params)
        recalls.append(recall_at_k(approx_ids, exact_ids))
    return round(statistics.fmean(recalls), 4)


# The btree indexes over the default filter columns. Dropping them (inside the
# rolled-back transaction) is the only way to force the HNSW index to be the
# access path for a filtered `ORDER BY <=> LIMIT k`: with them present the
# planner reads the partition by btree and sorts it exactly, and never touches
# the HNSW index -- which is itself a finding worth recording.
_FILTER_INDEXES = (
    "ix_chunk_susep_process",
    "ix_chunk_cnpj",
    "ix_chunk_susep_process_cnpj",
    "ix_chunk_clause_type",
)


def _reset_planner_gucs(session: Session) -> None:
    for guc in ("enable_seqscan", "enable_indexscan", "enable_indexonlyscan"):
        session.execute(text(f"RESET {guc}"))


def measure_filtered_counts(
    session: Session,
    *,
    query_vector: Sequence[float],
    filters: list[tuple[str, str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Count rows returned for each filter, across access paths.

    For each filter: the exact count, the planner's natural choice, and the
    HNSW-forced results with and without ``iterative_scan``. Runs last -- it
    drops the filter btree indexes to force the HNSW path, and the whole
    benchmark transaction is rolled back afterwards.
    """
    results: list[dict[str, Any]] = []
    literal = vector_literal(query_vector)

    for label, where, extra in filters:
        sql = f"SELECT chunk_id FROM chunk WHERE {where} {_ORDER_BY_DISTANCE}"
        params = {**extra, "q": literal, "k": K}

        _reset_planner_gucs(session)
        natural_plan = explain_scan(session, sql, params)
        natural = len(_topk_ids(session, sql, params))

        session.execute(text("SET LOCAL enable_indexscan = off"))
        session.execute(text("SET LOCAL enable_indexonlyscan = off"))
        exact = len(_topk_ids(session, sql, params))
        _reset_planner_gucs(session)
        results.append(
            {
                "filter": label,
                "exact": exact,
                "planner_default": natural,
                "planner_default_plan": natural_plan,
            }
        )

    # Force the HNSW path and re-measure every filter.
    for index_name in _FILTER_INDEXES:
        session.execute(text(f"DROP INDEX IF EXISTS {index_name}"))
    session.execute(text("SET LOCAL enable_seqscan = off"))

    for row, (_, where, extra) in zip(results, filters, strict=True):
        sql = f"SELECT chunk_id FROM chunk WHERE {where} {_ORDER_BY_DISTANCE}"
        params = {**extra, "q": literal, "k": K}
        apply_ann_search_gucs(session, iterative_scan="off")
        row["hnsw_forced_no_iterative_scan"] = len(_topk_ids(session, sql, params))
        row["hnsw_forced_plan"] = explain_scan(session, sql, params)
        apply_ann_search_gucs(session, iterative_scan="strict_order")
        row["hnsw_forced_strict_order"] = len(_topk_ids(session, sql, params))

    return results


# --------------------------------------------------------------------------- #
# Report                                                                     #
# --------------------------------------------------------------------------- #


class AnnBenchmarkConfig(BaseModel):
    """The reproducibility stamp for one ``make benchmark-ann-index`` run."""

    schema_version: str
    run_at_utc: datetime
    corpus_path: str
    chunk_count: int
    partition_count: int
    smallest_partition_size: int
    median_partition_size: int
    largest_partition_size: int
    hnsw_m: int
    hnsw_ef_construction: int
    hnsw_ef_search: int
    synthetic_vectors: bool
    vector_seed: int
    query_seed: int
    query_count: int
    measure_iterations: int
    k: int
    platform: str
    pgvector_version: str
    postgres_version: str


def build_report(
    config: AnnBenchmarkConfig,
    *,
    build: dict[str, Any],
    latency: dict[str, Any],
    ann_recall: float,
    filtered: list[dict[str, Any]],
) -> dict[str, Any]:
    """Assemble the single dict both the JSON and Markdown outputs render from."""
    return {
        "config": config.model_dump(mode="json"),
        "caveat": _CAVEAT,
        "build": build,
        "latency": latency,
        "ann_recall_at_k": ann_recall,
        "filtered_result_counts": filtered,
    }


def _latency_row(label: str, entry: dict[str, Any]) -> str:
    stats = entry["latency_ms"]
    return (
        f"| {label} | {stats['p50']:.3f} | {stats['p95']:.3f} | "
        f"{stats['mean']:.3f} | `{entry['plan']}` |"
    )


def render_markdown_report(report: dict[str, Any]) -> str:
    """Render the benchmark report as Markdown (numbers copied into EMBEDDINGS.md)."""
    config = report["config"]
    build = report["build"]
    latency = report["latency"]
    lines = [
        "# ANN index benchmark",
        "",
        "Generated by `scripts/benchmark_ann_index.py` (`make benchmark-ann-index`) "
        "against the built chunk corpus. Regenerable; the committed numbers and the "
        "verdict live in `docs/EMBEDDINGS.md`.",
        "",
        report["caveat"],
        "",
        "## Run configuration",
        "",
        f"- Corpus: `{config['corpus_path']}` ({config['chunk_count']} chunks, "
        f"{config['partition_count']} `(susep_process, cnpj)` partitions; sizes "
        f"{config['smallest_partition_size']} / {config['median_partition_size']} / "
        f"{config['largest_partition_size']} min/median/max)",
        f"- HNSW: `m={config['hnsw_m']}`, `ef_construction="
        f"{config['hnsw_ef_construction']}`, `ef_search={config['hnsw_ef_search']}`",
        f"- Queries: {config['query_count']} synthetic vectors x "
        f"{config['measure_iterations']} iterations; k={config['k']}",
        f"- pgvector {config['pgvector_version']}; {config['postgres_version']}",
        f"- Platform: {config['platform']}",
        f"- Run at (UTC): {config['run_at_utc']}",
        "",
        "## Build time and index size",
        "",
        f"- `CREATE INDEX` wall time: **{build['build_seconds']:.3f} s**",
        f"- Index size: **{build['index_size_pretty']}** "
        f"({build['index_bytes']} bytes)",
        f"- Table size: {build['table_bytes']} bytes; index/table ratio "
        f"{build['index_to_table_ratio']}",
        "",
        "## Latency: exact scan vs. HNSW",
        "",
        "| scope | p50 ms | p95 ms | mean ms | plan |",
        "| --- | ---: | ---: | ---: | --- |",
        _latency_row("exact, full corpus (no index)", latency["exact_full"]),
        _latency_row("exact, single partition (no index)", latency["exact_partition"]),
        _latency_row("with HNSW index, full corpus", latency["indexed_full"]),
        _latency_row("with HNSW index, single partition", latency["indexed_partition"]),
        "",
        f"ANN recall@{config['k']} vs. exact (synthetic vectors -- does not "
        f"transfer): **{report['ann_recall_at_k']:.4f}**",
        "",
        "## Filtered search: rows returned (k = " + str(config["k"]) + ")",
        "",
        "`exact` is the ground truth (index scans disabled). `planner default` "
        "is what the planner picks with every index available. `HNSW forced` "
        "drops the filter btree indexes and disables seq scan, so the HNSW "
        "index is the only access path -- the case `hnsw.iterative_scan` "
        "addresses.",
        "",
        "| filter | exact | planner default (plan) | HNSW forced, no iter. scan "
        "(plan) | HNSW forced, strict_order |",
        "| --- | ---: | --- | --- | ---: |",
    ]
    for row in report["filtered_result_counts"]:
        lines.append(
            f"| {row['filter']} | {row['exact']} "
            f"| {row['planner_default']} (`{row['planner_default_plan']}`) "
            f"| {row['hnsw_forced_no_iterative_scan']} "
            f"(`{row['hnsw_forced_plan']}`) "
            f"| {row['hnsw_forced_strict_order']} |"
        )
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Entry point                                                                #
# --------------------------------------------------------------------------- #


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-nontest-db",
        action="store_true",
        help="run even if the target database name has no 'test' in it",
    )
    return parser.parse_args()


def load_records() -> list[ChunkRecord]:
    """Load the built chunk corpus, failing loudly if it is not built yet."""
    if not CHUNKS_JSONL_PATH.exists():
        raise FileNotFoundError(
            f"{CHUNKS_JSONL_PATH} does not exist. Run `make build-chunks` (full "
            "rebuild) or `make fetch-corpus-artifacts` (pre-built corpus) first."
        )
    return read_chunks_jsonl(CHUNKS_JSONL_PATH)


def main() -> None:
    """Run the benchmark against the test database and write the report."""
    args = _parse_args()
    url = resolve_test_database_url()
    assert_target_is_test_db(url, allow_nontest=args.allow_nontest_db)
    records = load_records()
    smallest, median = choose_partitions(records)
    largest = max(Counter((r.susep_process, r.cnpj) for r in records).values())
    stacked = choose_stacked_clause_type(records, smallest)
    query_vectors = generate_query_vectors(QUERY_COUNT, QUERY_SEED)

    engine = create_engine_from_database_url(url)
    session = create_session_factory(engine=engine)()
    try:
        assert_chunk_table_ready(session)
        print(f"benchmark: loading {len(records)} chunks into {url.rsplit('/', 1)[-1]}")
        load_and_embed(session, records, seed=VECTOR_SEED)

        # Phase 1 -- exact, no index.
        exact_full = measure_latency(
            session, where="", base_params={}, query_vectors=query_vectors
        )
        exact_partition = measure_latency(
            session,
            where=_DEFAULT_FILTER,
            base_params=median.params,
            query_vectors=query_vectors,
        )

        # Phase 2 -- build the index, measure again.
        build = build_and_measure_index(session)
        apply_ann_search_gucs(session)
        indexed_full = measure_latency(
            session, where="", base_params={}, query_vectors=query_vectors
        )
        indexed_partition = measure_latency(
            session,
            where=_DEFAULT_FILTER,
            base_params=median.params,
            query_vectors=query_vectors,
        )

        ann_recall = measure_recall(session, query_vectors)

        filters: list[tuple[str, str, dict[str, Any]]] = [
            (
                f"susep_process + cnpj (smallest partition, {smallest.size} chunks)",
                _DEFAULT_FILTER,
                dict(smallest.params),
            )
        ]
        if stacked is not None:
            clause_type, count = stacked
            filters.append(
                (
                    f"+ clause_type = '{clause_type}' ({count} chunks)",
                    f"{_DEFAULT_FILTER} AND clause_type = :clause_type",
                    {**smallest.params, "clause_type": clause_type},
                )
            )
        filtered = measure_filtered_counts(
            session, query_vector=query_vectors[0], filters=filters
        )

        config = AnnBenchmarkConfig(
            schema_version=SCHEMA_VERSION,
            run_at_utc=datetime.now(UTC),
            corpus_path=str(CHUNKS_JSONL_PATH),
            chunk_count=len(records),
            partition_count=len(Counter((r.susep_process, r.cnpj) for r in records)),
            smallest_partition_size=smallest.size,
            median_partition_size=median.size,
            largest_partition_size=largest,
            hnsw_m=HNSW_M,
            hnsw_ef_construction=HNSW_EF_CONSTRUCTION,
            hnsw_ef_search=HNSW_EF_SEARCH,
            synthetic_vectors=True,
            vector_seed=VECTOR_SEED,
            query_seed=QUERY_SEED,
            query_count=QUERY_COUNT,
            measure_iterations=MEASURE_ITERATIONS,
            k=K,
            platform=platform.platform(),
            **server_versions(session),
        )
        report = build_report(
            config,
            build=build,
            latency={
                "exact_full": exact_full,
                "exact_partition": exact_partition,
                "indexed_full": indexed_full,
                "indexed_partition": indexed_partition,
            },
            ann_recall=ann_recall,
            filtered=filtered,
        )
    finally:
        session.rollback()
        session.close()
        engine.dispose()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    MD_PATH.write_text(render_markdown_report(report), encoding="utf-8")

    print(
        f"build {build['build_seconds']:.3f}s, index {build['index_size_pretty']}; "
        f"exact full p50 {exact_full['latency_ms']['p50']:.3f}ms, "
        f"indexed full p50 {indexed_full['latency_ms']['p50']:.3f}ms"
    )
    print(f"Wrote {JSON_PATH} and {MD_PATH}")


if __name__ == "__main__":
    main()
