"""The one verdict vocabulary lives in the domain layer [M4-01]."""

import pytest

from domain.verdict import Verdict


@pytest.mark.unit
def test_verdict_has_exactly_the_three_permitted_outcomes() -> None:
    assert {v.value for v in Verdict} == {
        "compatible",
        "incompatible",
        "insufficient_information",
    }


@pytest.mark.unit
def test_expected_verdict_is_an_alias_of_verdict() -> None:
    # The evaluation schemas and the M2 draft scripts still import
    # ``ExpectedVerdict`` from golden_set_schema; it must be the same object,
    # so golden-set-v1 JSONL and its validation are unaffected by the move.
    from infrastructure.evaluation.golden_set_schema import ExpectedVerdict

    assert ExpectedVerdict is Verdict
