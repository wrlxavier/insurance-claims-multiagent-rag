"""The end-to-end eval's own logic, offline [M4-10].

The live run is `make eval-end-to-end`; what is pinned here is the reasoning
the runner does *around* the graph — the failure catalogue's attribution order,
the policy header, the snapshot the CI gate consumes, and the aggregation. All
pure functions over constructed rows: no graph, no models, no database.

The catalogue's order is the part most worth a test. It is a causal order, not
an arbitrary one, and getting it wrong would silently misattribute whole
classes of failure — the exact thing the DoD says matters more than the
headline number.
"""

from typing import Any

import pytest
from scripts.eval_end_to_end import (
    _ClaimResult,
    _summarise,
    build_claim_text,
    failure_cause,
    render_markdown,
    snapshot_payload,
)


def _row(**overrides: Any) -> _ClaimResult:
    base: dict[str, Any] = {
        "claim_id": "compatible-001",
        "cohort": "compatible",
        "document_id": "1",
        "expected_verdict": "compatible",
        "predicted_verdict": "compatible",
        "posture": "compatible",
        "confidence": 0.7,
        "reached_checkpoint": True,
        "completed": True,
        "context_sufficient": True,
        "clarification_rounds": 0,
        "clarification_exhausted": False,
        "compatibility_verdict": "compatible",
        "grounding_degraded": False,
        "justification_degraded": False,
        "justification_names_a_clause": True,
        "assertions": (("O evento é coberto.", ("1:a",)),),
        "retrieved_clause_ids": ("1:a", "1:b"),
        "recommendation_citation_ids": ("1:a",),
        "reference_clause_ids": ("1:a",),
        "n_consistency_flags": 0,
        "latency_seconds": 42.0,
        "error": None,
    }
    base.update(overrides)
    return _ClaimResult(**base)


class _Claim:
    """The two SyntheticClaim fields build_claim_text reads."""

    def __init__(self, narrative: str) -> None:
        self.narrative = narrative


# --- the policy header -----------------------------------------------------


@pytest.mark.unit
def test_the_policy_header_states_the_process_and_keeps_the_narrative() -> None:
    text = build_claim_text(
        _Claim("bati o carro"),  # type: ignore[arg-type]
        {"susep_process": "15414.610650/2024-59"},
        policy_header=True,
    )
    assert text.startswith("[Apólice registrada: processo SUSEP 15414.610650/2024-59]")
    assert text.endswith("bati o carro")


@pytest.mark.unit
def test_the_header_adds_nothing_about_the_event() -> None:
    # It must carry the policy and only the policy: anything about the event
    # would be the harness answering the question it is scoring.
    row = {"susep_process": "15414.610650/2024-59", "product_line": "RCF-A"}
    text = build_claim_text(_Claim("n"), row, policy_header=True)  # type: ignore[arg-type]
    assert "RCF-A" not in text


@pytest.mark.unit
def test_the_no_header_arm_passes_the_bare_narrative() -> None:
    text = build_claim_text(
        _Claim("bati o carro"),  # type: ignore[arg-type]
        {"susep_process": "15414.610650/2024-59"},
        policy_header=False,
    )
    assert text == "bati o carro"


# --- the failure catalogue -------------------------------------------------


@pytest.mark.unit
def test_a_correct_verdict_is_never_catalogued() -> None:
    assert failure_cause(_row()) is None


@pytest.mark.unit
def test_an_errored_run_is_not_catalogued_as_a_reasoning_failure() -> None:
    assert failure_cause(_row(predicted_verdict=None, error="boom")) is None


@pytest.mark.unit
def test_claimant_gaps_wins_over_retrieval_signals() -> None:
    # An exhausted clarification loop never reaches retrieval, so its empty
    # citation list is a consequence, not the cause. Testing retrieval first
    # would attribute every one of these to a retrieval miss.
    row = _row(
        expected_verdict="compatible",
        predicted_verdict="insufficient_information",
        posture="claimant_gaps",
        clarification_exhausted=True,
        context_sufficient=None,
        retrieved_clause_ids=(),
    )
    assert failure_cause(row) == "claimant_gaps"


@pytest.mark.unit
def test_retrieval_miss_wins_over_assessment_signals() -> None:
    # A starved assessment degraded because it had nothing to ground against.
    row = _row(
        expected_verdict="compatible",
        predicted_verdict="insufficient_information",
        posture="retrieval_miss",
        context_sufficient=False,
        grounding_degraded=True,
        retrieved_clause_ids=(),
    )
    assert failure_cause(row) == "retrieval_miss"


@pytest.mark.unit
def test_a_labelled_clause_never_retrieved_is_a_retrieval_miss() -> None:
    row = _row(
        expected_verdict="compatible",
        predicted_verdict="incompatible",
        posture="incompatible",
        retrieved_clause_ids=("9:x",),
        reference_clause_ids=("1:a",),
    )
    assert failure_cause(row) == "retrieval_miss"


@pytest.mark.unit
def test_grounding_degradation_with_the_clause_in_hand_is_a_parsing_error() -> None:
    row = _row(
        expected_verdict="compatible",
        predicted_verdict="insufficient_information",
        posture="inconclusive",
        grounding_degraded=True,
    )
    assert failure_cause(row) == "parsing_error"


@pytest.mark.unit
def test_the_right_clause_and_the_wrong_verdict_is_a_reasoning_error() -> None:
    row = _row(expected_verdict="compatible", predicted_verdict="incompatible")
    assert failure_cause(row) == "reasoning_error"


# --- the on-target-document diagnostic -------------------------------------


@pytest.mark.unit
def test_on_target_document_separates_wrong_document_from_wrong_clause() -> None:
    wrong_clause = _row(retrieved_clause_ids=("1:z",), reference_clause_ids=("1:a",))
    assert wrong_clause.on_target_document is True
    assert wrong_clause.reference_hits == 0

    wrong_document = _row(retrieved_clause_ids=("9:z",), reference_clause_ids=("1:a",))
    assert wrong_document.on_target_document is False


@pytest.mark.unit
def test_no_retrieval_is_not_on_target() -> None:
    assert _row(retrieved_clause_ids=()).on_target_document is False


# --- aggregation -----------------------------------------------------------


def _result(rows: list[_ClaimResult]) -> Any:
    return _summarise(rows, [], None, meta=_META)


_META: dict[str, Any] = {
    "generated_at_utc": "2026-09-02T00:00:00+00:00",
    "platform": "test",
    "reasoning_model": "r",
    "fast_model": "f",
    "judge_model": None,
    "judge_passes": None,
    "policy_header": True,
    "claim_count": 2,
    "dod_note": "note",
}


@pytest.mark.unit
def test_cohorts_are_scored_separately_including_mismatch() -> None:
    result = _result(
        [
            _row(),
            _row(
                claim_id="mismatch-001",
                cohort="mismatch",
                expected_verdict="incompatible",
                predicted_verdict="insufficient_information",
                posture="retrieval_miss",
                context_sufficient=False,
                retrieved_clause_ids=(),
            ),
        ]
    )
    assert set(result.by_cohort) == {"compatible", "mismatch"}
    assert result.by_cohort["mismatch"].accuracy == 0.0
    assert result.by_cohort["compatible"].accuracy == 1.0
    assert result.overall.accuracy == 0.5


@pytest.mark.unit
def test_the_catalogue_accounts_for_every_wrong_verdict_and_nothing_else() -> None:
    rows = [
        _row(),
        _row(
            claim_id="x1",
            expected_verdict="compatible",
            predicted_verdict="incompatible",
        ),
        _row(
            claim_id="x2",
            expected_verdict="compatible",
            predicted_verdict="insufficient_information",
            clarification_exhausted=True,
        ),
    ]
    result = _result(rows)
    assert sum(result.failure_causes.values()) == 2
    assert result.failure_causes["reasoning_error"] == 1
    assert result.failure_causes["claimant_gaps"] == 1


@pytest.mark.unit
def test_an_errored_claim_is_excluded_from_accuracy_but_counted_in_completion() -> None:
    rows = [
        _row(),
        _row(
            claim_id="e1",
            predicted_verdict=None,
            completed=False,
            reached_checkpoint=False,
            error="boom",
        ),
    ]
    result = _summarise(rows, ["e1"], None, meta=_META)
    assert result.overall.n == 1
    assert result.overall.accuracy == 1.0
    assert result.completion_rate == 0.5
    assert result.error_claim_ids == ["e1"]


# --- the committed snapshot ------------------------------------------------


@pytest.mark.unit
def test_the_snapshot_carries_the_assertions_and_their_clause_ids() -> None:
    payload = snapshot_payload(_result([_row()]))
    claim = payload["claims"][0]
    assert claim["claim_id"] == "compatible-001"
    assert claim["assertions"] == [
        {"statement": "O evento é coberto.", "clause_ids": ["1:a"]}
    ]
    assert claim["retrieved_clause_ids"] == ["1:a", "1:b"]
    assert payload["provenance"]["generated_by"] == "scripts/eval_end_to_end.py"


@pytest.mark.unit
def test_the_snapshot_omits_claims_the_run_errored_on() -> None:
    # A claim with no run behind it is not evidence about citation coverage,
    # and including it would make the gate assert over an empty assertion list.
    payload = snapshot_payload(
        _result([_row(), _row(claim_id="e1", predicted_verdict=None, error="boom")])
    )
    assert [c["claim_id"] for c in payload["claims"]] == ["compatible-001"]


# --- rendering -------------------------------------------------------------


@pytest.mark.unit
def test_the_report_renders_the_matrix_the_cohorts_and_the_catalogue() -> None:
    markdown = render_markdown(_result([_row()]))
    assert "# End-to-end verdict accuracy" in markdown
    assert "| expected \\ predicted |" in markdown
    assert "#### compatible" in markdown
    assert "## Failure catalogue" in markdown
    assert "| reasoning_error | 0 |" in markdown


@pytest.mark.unit
def test_the_report_names_the_mismatch_cohort_as_the_dod_calls_it() -> None:
    markdown = render_markdown(
        _result(
            [
                _row(
                    claim_id="mismatch-001",
                    cohort="mismatch",
                    expected_verdict="incompatible",
                    predicted_verdict="incompatible",
                    posture="incompatible",
                )
            ]
        )
    )
    assert "product/claim mismatch subset" in markdown


@pytest.mark.unit
def test_a_failed_judge_is_reported_not_hidden() -> None:
    # The judge runs after every claim is scored, so its failure must cost the
    # judge's metrics and nothing else -- and the report has to say so, or a
    # reader would take a missing section for a section never asked for.
    result = _summarise(
        [_row()], [], None, meta={**_META, "judge_error": "RuntimeError('502')"}
    )
    markdown = render_markdown(result)
    assert "The judge did not complete" in markdown
    assert "RuntimeError('502')" in markdown
    assert result.to_json()["judge"] is None
    # The verdict numbers survive.
    assert result.overall.accuracy == 1.0
