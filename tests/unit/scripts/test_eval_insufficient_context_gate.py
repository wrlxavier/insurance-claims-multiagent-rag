"""Tests for the [M3-07] insufficient-context gate calibration script."""

import pytest
from scripts.eval_insufficient_context_gate import (
    InsufficientContextGateConfig,
    QuestionSignals,
    build_report,
    confusion_at,
    fragility,
    operating_point,
    pure_floor_sweep,
    render_markdown_report,
)

from infrastructure.rag.insufficient_context_config import (
    INSTANCE_VALUE_TOP_SCORE_THRESHOLD,
    TOP_SCORE_ABSTAIN_THRESHOLD,
)

# A "rule question" about a deductible -- names franquia, asks "de que forma";
# `needs_verified_instance_value` is False, so only the primary floor applies.
_RULE_Q = "De que forma o valor da franquia é estruturado no seguro?"
# An "instance-value question" -- names franquia, asks for "o valor exato";
# `needs_verified_instance_value` is True, so the strict floor applies.
_VALUE_Q = "Qual é o valor exato da franquia contratada para o veículo segurado?"


def _row(
    question_id: str,
    *,
    unanswerable: bool,
    top_score: float,
    question: str = _RULE_Q,
    n_returned: int = 10,
) -> QuestionSignals:
    return QuestionSignals(
        question_id=question_id,
        question_type="unanswerable" if unanswerable else "direct_lookup",
        question=question,
        is_unanswerable=unanswerable,
        top_score=top_score,
        reranked_scores=[top_score] * n_returned,
        retrieved_clause_ids=[f"1:c{i}" for i in range(n_returned)],
        retrieved_clause_types=["coverage"] * n_returned,
        n_returned=n_returned,
        k_requested=10,
    )


@pytest.mark.unit
def test_confusion_at_counts_abstentions_by_class() -> None:
    rows = [
        _row("unanswerable-001", unanswerable=True, top_score=0.20),
        _row("unanswerable-002", unanswerable=True, top_score=0.55, question=_VALUE_Q),
        _row("direct_lookup-001", unanswerable=False, top_score=0.90),
        _row("direct_lookup-002", unanswerable=False, top_score=0.35),  # a FP
    ]
    result = confusion_at(rows, low=0.40, high=0.84)
    assert result["tp"] == 2  # u-001 via floor, u-002 via instance-value rule
    assert result["fn"] == 0
    assert result["fp"] == 1  # dl-002 at 0.35 < 0.40 floor
    assert result["tn"] == 1
    assert result["recall"] == 1.0
    assert result["precision"] == pytest.approx(2 / 3)
    assert result["fp_ids"] == ["direct_lookup-002 (direct_lookup)"]


@pytest.mark.unit
def test_instance_value_rule_is_disabled_when_high_is_zero() -> None:
    rows = [
        _row("unanswerable-002", unanswerable=True, top_score=0.55, question=_VALUE_Q),
    ]
    # high=0 -> rule (c) never fires; 0.55 >= 0.40 floor -> not abstained -> FN.
    assert confusion_at(rows, low=0.40, high=0.0)["fn"] == 1
    # rule (c) on -> 0.55 < 0.84 and it is an instance-value question -> abstained.
    assert confusion_at(rows, low=0.40, high=0.84)["tp"] == 1


@pytest.mark.unit
def test_pure_floor_sweep_disables_rule_c_and_includes_the_full_recall_point() -> None:
    rows = [
        _row("unanswerable-001", unanswerable=True, top_score=0.30),
        _row("unanswerable-002", unanswerable=True, top_score=0.80, question=_VALUE_Q),
        _row("direct_lookup-001", unanswerable=False, top_score=0.95),
    ]
    curve = pure_floor_sweep(rows)
    assert all(point["high"] == 0.0 for point in curve)  # rule (c) off everywhere
    # A row just above the highest unanswerable top-score -> 100% recall.
    full_recall = [p for p in curve if p["recall"] == 1.0]
    assert full_recall and min(p["low"] for p in full_recall) == pytest.approx(
        0.801, abs=1e-3
    )


@pytest.mark.unit
def test_operating_point_attributes_each_catch_to_a_rule() -> None:
    rows = [
        _row("unanswerable-001", unanswerable=True, top_score=0.20),  # via floor
        _row(
            "unanswerable-002", unanswerable=True, top_score=0.70, question=_VALUE_Q
        ),  # via instance-value rule
        _row("direct_lookup-001", unanswerable=False, top_score=0.95),
    ]
    op = operating_point(rows)
    assert op["caught_by_low_floor"] == ["unanswerable-001"]
    assert op["caught_by_instance_value_rule"] == [("unanswerable-002", 0.7)]
    assert op["unanswerable_missed"] == []
    assert op["recall"] == 1.0
    assert op["fp"] == 0


@pytest.mark.unit
def test_fragility_reports_both_threshold_gaps() -> None:
    rows = [
        _row("unanswerable-001", unanswerable=True, top_score=0.30),  # floor-side
        _row(
            "unanswerable-002", unanswerable=True, top_score=0.80, question=_VALUE_Q
        ),  # instance-value-side
        _row("direct_lookup-001", unanswerable=False, top_score=0.62),
        _row(
            "direct_lookup-002", unanswerable=False, top_score=0.91, question=_VALUE_Q
        ),
    ]
    frag = fragility(rows)
    assert frag["low_floor"] == TOP_SCORE_ABSTAIN_THRESHOLD
    assert frag["low_floor_gap"] == [0.3, 0.62]
    assert frag["instance_value_floor"] == INSTANCE_VALUE_TOP_SCORE_THRESHOLD
    assert frag["instance_value_floor_gap"] == [0.8, 0.91]


@pytest.mark.unit
def test_render_markdown_report_has_the_key_sections() -> None:
    rows = [
        _row("unanswerable-001", unanswerable=True, top_score=0.20),
        _row("unanswerable-002", unanswerable=True, top_score=0.60, question=_VALUE_Q),
        _row("direct_lookup-001", unanswerable=False, top_score=0.85),
    ]
    config = InsufficientContextGateConfig(
        schema_version="v1",
        run_at_utc="2026-08-30T00:00:00+00:00",
        golden_set_dir="data/golden_set",
        golden_set_question_count=140,
        unanswerable_count=23,
        scorable_count=117,
        corpus_path="build/parsed_clauses.jsonl",
        corpus_clause_count=4925,
        rerank_candidate_depth=10,
        reranker_model_id="Alibaba-NLP/gte-multilingual-reranker-base",
        reranker_model_revision="8215cf04",
        reranker_config_fingerprint="777c0503f1073d52",
        embedding_config_fingerprint="7ea39a621eaee88e",
        lexical_config_fingerprint="ef0a2dd0c1dfb4e4",
        hybrid_config_fingerprint="279ed8ee0a668227",
        gate_config_fingerprint="deadbeefdeadbeef",
        chosen_top_score_threshold=0.46,
        chosen_instance_value_threshold=0.84,
        scoring_device="cuda:0",
        platform="Linux",
    )
    markdown = render_markdown_report(build_report(rows, config))

    assert "# Insufficient-context gate calibration" in markdown
    assert "## The rule" in markdown
    assert "## Can a pure top-score gate work?" in markdown
    assert "## Operating point" in markdown
    assert "## Calibration fragility" in markdown
    assert "deadbeefdeadbeef" in markdown
