"""Round-trip test for the retrieval-run config stamp [M2-06]."""

from datetime import UTC, datetime

import pytest

from infrastructure.evaluation.retrieval_run_schema import (
    SCHEMA_VERSION,
    RetrievalRunConfig,
)


@pytest.mark.unit
def test_retrieval_run_config_json_round_trips() -> None:
    config = RetrievalRunConfig(
        schema_version=SCHEMA_VERSION,
        retriever_name="random",
        k_values=[1, 5, 10],
        ndcg_k=10,
        golden_set_dir="data/golden_set",
        golden_set_question_count=140,
        corpus_path="build/parsed_clauses.jsonl",
        corpus_clause_count=4925,
        seed=42,
        run_at_utc=datetime.now(UTC),
    )

    restored = RetrievalRunConfig.model_validate_json(config.model_dump_json())

    assert restored == config


@pytest.mark.unit
def test_lexical_fields_are_optional_and_round_trip() -> None:
    # `random` leaves every [M3-03] field None; `lexical` sets them.
    base = {
        "schema_version": SCHEMA_VERSION,
        "retriever_name": "lexical",
        "k_values": [1, 5, 10],
        "ndcg_k": 10,
        "golden_set_dir": "data/golden_set",
        "golden_set_question_count": 140,
        "corpus_path": "build/parsed_clauses.jsonl",
        "corpus_clause_count": 4925,
        "seed": None,
        "run_at_utc": datetime.now(UTC),
    }
    assert RetrievalRunConfig(**base).chunk_corpus_path is None

    config = RetrievalRunConfig(
        **base,
        chunk_corpus_path="build/chunks.jsonl",
        chunk_corpus_chunk_count=4540,
        lexical_analyzer_version="v1",
        bm25_k1=1.5,
        bm25_b=0.75,
        lexical_idf_variant="lucene_plus_one",
        lexical_index_text_field="text",
        stemming_exception_count=5,
        lexical_config_fingerprint="deadbeefdeadbeef",
    )
    assert RetrievalRunConfig.model_validate_json(config.model_dump_json()) == config
