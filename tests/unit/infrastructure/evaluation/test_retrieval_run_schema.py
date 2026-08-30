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


@pytest.mark.unit
def test_v3_filter_and_fusion_fields_are_optional_and_round_trip() -> None:
    # `random`/`lexical` leave every [M3-04] field None; `hybrid` sets them.
    base = {
        "schema_version": SCHEMA_VERSION,
        "retriever_name": "hybrid",
        "k_values": [1, 5, 10],
        "ndcg_k": 10,
        "golden_set_dir": "data/golden_set",
        "golden_set_question_count": 140,
        "corpus_path": "build/parsed_clauses.jsonl",
        "corpus_clause_count": 4925,
        "seed": None,
        "run_at_utc": datetime.now(UTC),
    }
    assert RetrievalRunConfig(**base).fusion_strategy is None

    config = RetrievalRunConfig(
        **base,
        filter_mode="default",
        dense_model_id="Alibaba-NLP/gte-multilingual-base",
        dense_model_revision="9bbca17d",
        embedding_config_fingerprint="7ea39a621eaee88e",
        fusion_strategy="rrf",
        rrf_k=60,
        fusion_weights=[0.5, 0.5],
        candidate_depth=100,
        hybrid_config_fingerprint="279ed8ee0a668227",
    )
    assert RetrievalRunConfig.model_validate_json(config.model_dump_json()) == config


@pytest.mark.unit
def test_v4_rerank_fields_are_optional_and_round_trip() -> None:
    # Every non-rerank run leaves the [M3-05] fields None; `--rerank` sets them.
    base = {
        "schema_version": SCHEMA_VERSION,
        "retriever_name": "hybrid",
        "k_values": [1, 5, 10],
        "ndcg_k": 10,
        "golden_set_dir": "data/golden_set",
        "golden_set_question_count": 140,
        "corpus_path": "build/parsed_clauses.jsonl",
        "corpus_clause_count": 4925,
        "seed": None,
        "run_at_utc": datetime.now(UTC),
    }
    assert RetrievalRunConfig(**base).reranker_model_id is None

    config = RetrievalRunConfig(
        **base,
        filter_mode="default",
        fusion_strategy="rrf",
        reranker_model_id="Alibaba-NLP/gte-multilingual-reranker-base",
        reranker_model_revision="8215cf04918ba6f7b6a62bb44238ce2953d8831c",
        rerank_candidate_depth=10,
        reranker_config_fingerprint="777c0503f1073d52",
    )
    assert RetrievalRunConfig.model_validate_json(config.model_dump_json()) == config


@pytest.mark.unit
def test_v5_co_retrieval_fields_are_optional_and_round_trip() -> None:
    # Every non-co-retrieval run leaves the [M3-06] fields None; the flag sets them.
    base = {
        "schema_version": SCHEMA_VERSION,
        "retriever_name": "hybrid",
        "k_values": [1, 5, 10],
        "ndcg_k": 10,
        "golden_set_dir": "data/golden_set",
        "golden_set_question_count": 140,
        "corpus_path": "build/parsed_clauses.jsonl",
        "corpus_clause_count": 4925,
        "seed": None,
        "run_at_utc": datetime.now(UTC),
    }
    assert RetrievalRunConfig(**base).reserved_exclusion_slots is None

    config = RetrievalRunConfig(
        **base,
        filter_mode="default",
        fusion_strategy="rrf",
        reserved_exclusion_slots=2,
        adjacent_section_max_page_gap=3,
        co_retrieval_config_fingerprint="7ed4c97c4e8f1cb4",
    )
    assert RetrievalRunConfig.model_validate_json(config.model_dump_json()) == config
