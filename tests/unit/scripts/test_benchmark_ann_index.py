"""Unit tests for the ANN index benchmark script -- [M3-02].

Pure functions only -- the database-backed measurement is exercised by hand via
``make benchmark-ann-index`` and the mechanism it characterises is proven in
``tests/integration/test_ann_index.py``.
"""

import math
from datetime import UTC, datetime

import pytest
from scripts.benchmark_ann_index import (
    AnnBenchmarkConfig,
    K,
    Partition,
    _parse_args,
    _scan_summary,
    build_report,
    choose_partitions,
    choose_stacked_clause_type,
    generate_query_vectors,
    pseudo_random_unit_vector,
    recall_at_k,
    render_markdown_report,
    render_real_report,
    summarise_latency,
    vector_literal,
)

from infrastructure.rag.chunk_schema import SCHEMA_VERSION, ChunkRecord
from infrastructure.rag.embedding_config import EMBEDDING_DIMENSIONS


def _record(chunk_id: str, *, susep: str, cnpj: str, clause_type: str) -> ChunkRecord:
    return ChunkRecord.model_validate(
        {
            "schema_version": SCHEMA_VERSION,
            "chunk_id": chunk_id,
            "document_id": "1",
            "clause_id": chunk_id,
            "source_clause_ids": [chunk_id],
            "chunk_index": 0,
            "chunk_count": 1,
            "parent_path": "",
            "text": "Texto.",
            "display_text": "Texto.",
            "char_count": 6,
            "rule": "single",
            "clause_type": clause_type,
            "type_source": "rule",
            "confidence": None,
            "bundle_section": None,
            "source": "text",
            "susep_process": susep,
            "insurer": "Seguradora",
            "cnpj": cnpj,
            "product_line": "CASCO",
            "indemnity_regime": "VD",
            "filing_year": "2020",
        }
    )


@pytest.mark.unit
def test_pseudo_random_unit_vector_is_deterministic() -> None:
    assert pseudo_random_unit_vector("k") == pseudo_random_unit_vector("k")
    assert pseudo_random_unit_vector("a") != pseudo_random_unit_vector("b")


@pytest.mark.unit
def test_pseudo_random_unit_vector_is_unit_norm_and_the_pinned_width() -> None:
    vector = pseudo_random_unit_vector("chunk:1")

    assert len(vector) == EMBEDDING_DIMENSIONS
    assert math.sqrt(sum(value * value for value in vector)) == pytest.approx(1.0)


@pytest.mark.unit
def test_pseudo_random_unit_vector_is_sign_varied() -> None:
    # The sha256-byte FakeEmbedder is all-positive; this one must not be.
    vector = pseudo_random_unit_vector("chunk:1")

    assert min(vector) < 0.0 < max(vector)


@pytest.mark.unit
def test_distinct_pseudo_random_vectors_are_not_near_parallel() -> None:
    a = pseudo_random_unit_vector("chunk:1")
    b = pseudo_random_unit_vector("chunk:2")

    dot = sum(x * y for x, y in zip(a, b, strict=True))
    assert abs(dot) < 0.3


@pytest.mark.unit
def test_generate_query_vectors_is_reproducible_and_sized() -> None:
    first = generate_query_vectors(5, seed=7)
    second = generate_query_vectors(5, seed=7)

    assert first == second
    assert len(first) == 5
    assert generate_query_vectors(5, seed=8) != first


@pytest.mark.unit
def test_vector_literal_round_trips_the_components() -> None:
    assert vector_literal([1.0, -0.5, 0.25]) == "[1.0,-0.5,0.25]"


@pytest.mark.unit
def test_recall_at_k() -> None:
    assert recall_at_k(["a", "x"], ["a", "b"]) == 0.5
    assert recall_at_k(["a", "b", "c"], ["a", "b"]) == 1.0
    assert recall_at_k(["x"], ["a", "b"]) == 0.0
    assert recall_at_k([], []) == 1.0  # nothing to miss


@pytest.mark.unit
def test_summarise_latency_percentiles() -> None:
    summary = summarise_latency([float(value) for value in range(1, 21)])

    assert summary["n"] == 20
    assert summary["p50"] == 11.0
    assert summary["p95"] == 20.0
    assert summary["mean"] == 10.5


@pytest.mark.unit
def test_summarise_latency_handles_no_samples() -> None:
    assert summarise_latency([]) == {"n": 0, "p50": 0.0, "p95": 0.0, "mean": 0.0}


@pytest.mark.unit
def test_choose_partitions_returns_smallest_and_median() -> None:
    def block(prefix: str, susep: str, cnpj: str, count: int) -> list[ChunkRecord]:
        return [
            _record(f"{prefix}{i}", susep=susep, cnpj=cnpj, clause_type="coverage")
            for i in range(count)
        ]

    records = (
        block("s", "p-small", "c1", 2)
        + block("m", "p-med", "c2", 5)
        + block("l", "p-large", "c3", 9)
    )

    smallest, median = choose_partitions(records)

    assert (smallest.susep_process, smallest.size) == ("p-small", 2)
    assert (median.susep_process, median.size) == ("p-med", 5)


@pytest.mark.unit
def test_choose_stacked_clause_type_picks_an_under_k_type() -> None:
    partition = Partition("p", "c", 12)
    records = [
        _record(f"e{i}", susep="p", cnpj="c", clause_type="exclusion") for i in range(3)
    ] + [
        _record(f"cov{i}", susep="p", cnpj="c", clause_type="coverage")
        for i in range(K + 2)
    ]

    result = choose_stacked_clause_type(records, partition)

    assert result == ("exclusion", 3)


@pytest.mark.unit
def test_choose_stacked_clause_type_returns_none_when_every_type_fills_k() -> None:
    partition = Partition("p", "c", 40)
    records = [
        _record(f"cov{i}", susep="p", cnpj="c", clause_type="coverage")
        for i in range(K + 5)
    ] + [
        _record(f"exc{i}", susep="p", cnpj="c", clause_type="exclusion")
        for i in range(K + 5)
    ]

    assert choose_stacked_clause_type(records, partition) is None


@pytest.mark.unit
def test_scan_summary_finds_the_named_index_scan() -> None:
    plan = {
        "Node Type": "Limit",
        "Plans": [{"Node Type": "Index Scan", "Index Name": "ix_chunk_embedding_hnsw"}],
    }

    assert _scan_summary(plan) == "Index Scan using ix_chunk_embedding_hnsw"


@pytest.mark.unit
def test_scan_summary_reports_a_bare_seq_scan() -> None:
    assert _scan_summary({"Node Type": "Seq Scan"}) == "Seq Scan"


def _config() -> AnnBenchmarkConfig:
    return AnnBenchmarkConfig(
        schema_version="v1",
        run_at_utc=datetime(2026, 8, 28, tzinfo=UTC),
        corpus_path="build/chunks.jsonl",
        chunk_count=4540,
        partition_count=30,
        smallest_partition_size=19,
        median_partition_size=109,
        largest_partition_size=726,
        hnsw_m=16,
        hnsw_ef_construction=64,
        hnsw_ef_search=40,
        synthetic_vectors=True,
        vector_seed=1,
        query_seed=2,
        query_count=50,
        measure_iterations=10,
        k=10,
        platform="Linux",
        pgvector_version="0.8.6",
        postgres_version="PostgreSQL 17",
    )


def _latency_entry() -> dict[str, object]:
    return {
        "latency_ms": {"n": 500, "p50": 0.8, "p95": 1.2, "mean": 0.9},
        "plan": "Seq Scan",
    }


@pytest.mark.unit
def test_render_markdown_report_has_every_section_and_the_caveat() -> None:
    report = build_report(
        _config(),
        build={
            "build_seconds": 0.79,
            "index_bytes": 9306112,
            "index_size_pretty": "9088 kB",
            "table_bytes": 13533184,
            "index_to_table_ratio": 0.688,
        },
        latency={
            "exact_full": _latency_entry(),
            "exact_partition": _latency_entry(),
            "indexed_full": _latency_entry(),
            "indexed_partition": _latency_entry(),
        },
        ann_recall=0.45,
        filtered=[
            {
                "filter": "susep_process + cnpj (19 chunks)",
                "exact": 10,
                "planner_default": 10,
                "planner_default_plan": "Index Scan using ix_chunk_susep_process_cnpj",
                "hnsw_forced_no_iterative_scan": 0,
                "hnsw_forced_plan": "Index Scan using ix_chunk_embedding_hnsw",
                "hnsw_forced_strict_order": 10,
            }
        ],
    )

    markdown = render_markdown_report(report)

    for header in (
        "# ANN index benchmark",
        "## Build time and index size",
        "## Latency: exact scan vs. HNSW",
        "## Filtered search",
    ):
        assert header in markdown
    assert "Synthetic vectors" in markdown
    assert "| --- | ---: | ---: | ---: | --- |" in markdown
    assert "9088 kB" in markdown


@pytest.mark.unit
def test_parse_args_exposes_the_real_embeddings_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("sys.argv", ["benchmark_ann_index.py"])
    assert _parse_args().real_embeddings is False
    monkeypatch.setattr("sys.argv", ["benchmark_ann_index.py", "--real-embeddings"])
    assert _parse_args().real_embeddings is True


@pytest.mark.unit
def test_render_real_report_states_the_recall_and_the_filtered_plan() -> None:
    config = _config().model_dump(mode="json")
    config["synthetic_vectors"] = False
    report = {
        "config": config,
        "caveat": "**Real embeddings.** ... rolled back ...",
        "embedding_config_fingerprint": "7ea39a621eaee88e",
        "build": {
            "build_seconds": 0.81,
            "index_bytes": 9306112,
            "index_size_pretty": "9088 kB",
            "table_bytes": 13533184,
            "index_to_table_ratio": 0.688,
        },
        "latency": {
            "exact_full": _latency_entry(),
            "exact_partition": _latency_entry(),
            "indexed_full": _latency_entry(),
            "indexed_partition": _latency_entry(),
        },
        "ann_recall_at_k_full_corpus": 0.9631,
        "filtered_path_plan": "Index Scan using ix_chunk_susep_process_cnpj",
    }

    markdown = render_real_report(report)

    assert "# ANN index benchmark -- real embeddings" in markdown
    assert "## ANN recall vs. exact" in markdown
    assert "0.9631" in markdown
    assert "ix_chunk_susep_process_cnpj" in markdown
    assert "recall 1.0 by construction" in markdown
    assert "7ea39a621eaee88e" in markdown
