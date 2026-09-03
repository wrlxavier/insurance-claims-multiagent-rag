"""Tests for the shared three-class verdict metrics [M4-10]."""

import pytest

from domain.verdict import Verdict
from infrastructure.evaluation.verdict_metrics import (
    VERDICTS,
    confusion_table_lines,
    metrics_json,
    per_class_table_lines,
    verdict_metrics,
)

_C = "compatible"
_I = "incompatible"
_U = "insufficient_information"


@pytest.mark.unit
def test_column_order_comes_from_the_verdict_enum() -> None:
    # Not a restated literal: a fourth Verdict would appear here automatically.
    assert VERDICTS == tuple(v.value for v in Verdict)


@pytest.mark.unit
def test_perfect_prediction() -> None:
    metrics = verdict_metrics([(_C, _C), (_I, _I), (_U, _U)])
    assert metrics.n == 3
    assert metrics.accuracy == 1.0
    for verdict in VERDICTS:
        assert metrics.per_class[verdict]["precision"] == 1.0
        assert metrics.per_class[verdict]["recall"] == 1.0


@pytest.mark.unit
def test_confusion_counts_expected_by_predicted() -> None:
    metrics = verdict_metrics([(_I, _U), (_I, _U), (_C, _C)])
    assert metrics.confusion[_I][_U] == 2
    assert metrics.confusion[_I][_I] == 0
    assert metrics.confusion[_C][_C] == 1
    assert metrics.accuracy == pytest.approx(1 / 3)


@pytest.mark.unit
def test_per_class_precision_and_recall() -> None:
    # incompatible: 1 tp, 1 fp (a compatible predicted incompatible), 1 fn.
    metrics = verdict_metrics([(_I, _I), (_I, _U), (_C, _I)])
    stats = metrics.per_class[_I]
    assert stats["support"] == 2.0
    assert stats["precision"] == pytest.approx(0.5)
    assert stats["recall"] == pytest.approx(0.5)


@pytest.mark.unit
def test_a_class_with_no_support_scores_zero_not_nan() -> None:
    metrics = verdict_metrics([(_C, _C)])
    assert metrics.per_class[_U]["support"] == 0.0
    assert metrics.per_class[_U]["recall"] == 0.0
    assert metrics.per_class[_U]["precision"] == 0.0


@pytest.mark.unit
def test_unpredicted_rows_are_excluded_not_counted_wrong() -> None:
    # An item the run errored on is not evidence about accuracy: excluding it
    # keeps the denominator honest, and the caller reports the exclusions.
    metrics = verdict_metrics([(_C, _C), (_I, None), (_U, None)])
    assert metrics.n == 1
    assert metrics.accuracy == 1.0


@pytest.mark.unit
def test_empty_population_is_zero_not_a_crash() -> None:
    metrics = verdict_metrics([])
    assert metrics.n == 0
    assert metrics.accuracy == 0.0


@pytest.mark.unit
def test_an_unknown_verdict_raises() -> None:
    with pytest.raises(ValueError, match="denied"):
        verdict_metrics([("denied", _C)])
    with pytest.raises(ValueError, match="covered"):
        verdict_metrics([(_C, "covered")])


@pytest.mark.unit
def test_markdown_tables_have_a_row_per_verdict() -> None:
    metrics = verdict_metrics([(_C, _C), (_I, _U)])
    confusion = confusion_table_lines(metrics)
    assert len(confusion) == 2 + len(VERDICTS)
    assert confusion[0].startswith("| expected \\ predicted |")
    assert f"| {_I} | 0 | 0 | 1 |" in confusion

    per_class = per_class_table_lines(metrics)
    assert len(per_class) == 2 + len(VERDICTS)
    assert per_class[0].startswith("| verdict |")


@pytest.mark.unit
def test_metrics_json_round_trips_the_dataclass() -> None:
    metrics = verdict_metrics([(_C, _C), (_I, _U)])
    payload = metrics_json(metrics)
    assert payload["n"] == 2
    assert payload["confusion"] == metrics.confusion
    assert payload["per_class"] == metrics.per_class
