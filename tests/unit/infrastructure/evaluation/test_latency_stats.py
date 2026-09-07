"""The shared nearest-rank latency helper [M5-10].

Pins the formula the four measurement scripts used to each keep a private copy
of, and the two rounding-precision paths their ``summarise_latency`` wrappers
need.
"""

import pytest

from infrastructure.evaluation.latency_stats import percentile, summarise_latency


def _old_percentile(sorted_values: list[float], fraction: float) -> float:
    """The formula as it stood in ``benchmark_ann_index`` before this module."""
    index = min(int(len(sorted_values) * fraction), len(sorted_values) - 1)
    return sorted_values[index]


@pytest.mark.unit
def test_percentile_matches_the_pre_consolidation_formula() -> None:
    values = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]
    ordered = sorted(values)
    for fraction in (0.0, 0.25, 0.5, 0.9, 0.95, 0.99, 1.0):
        assert percentile(values, fraction) == _old_percentile(ordered, fraction)


@pytest.mark.unit
def test_percentile_sorts_unsorted_input() -> None:
    assert percentile([30, 10, 20], 0.5) == 20


@pytest.mark.unit
def test_percentile_preserves_the_element_type() -> None:
    result = percentile([10, 20, 30, 40, 50], 0.5)
    assert result == 30
    assert isinstance(result, int)


@pytest.mark.unit
def test_percentile_single_value() -> None:
    assert percentile([7.5], 0.95) == 7.5


@pytest.mark.unit
def test_percentile_p95_of_fifty_one_is_the_forty_ninth() -> None:
    # nearest-rank: index = min(int(51 * 0.95), 50) = min(48, 50) = 48
    values = list(range(51))
    assert percentile(values, 0.95) == 48


@pytest.mark.unit
def test_percentile_raises_on_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        percentile([], 0.5)


@pytest.mark.unit
def test_summarise_latency_empty_is_all_zero() -> None:
    assert summarise_latency([]) == {"n": 0, "p50": 0.0, "p95": 0.0, "mean": 0.0}


@pytest.mark.unit
def test_summarise_latency_default_precision_is_three() -> None:
    # benchmark_ann_index's contract: 1..20, p50 = 11.0, p95 = 20.0.
    summary = summarise_latency([float(v) for v in range(1, 21)])
    assert summary == {"n": 20, "p50": 11.0, "p95": 20.0, "mean": 10.5}


@pytest.mark.unit
def test_summarise_latency_rounds_to_the_requested_digits() -> None:
    samples = [1.0, 1.0, 1.0, 100.0 / 3.0]
    assert summarise_latency(samples, digits=1)["mean"] == 9.1
    assert summarise_latency(samples, digits=3)["mean"] == 9.083
