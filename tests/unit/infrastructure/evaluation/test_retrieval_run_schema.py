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
