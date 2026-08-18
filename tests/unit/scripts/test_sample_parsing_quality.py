"""Tests for the [M1-08] parsing-quality sampling script."""

import pytest
from scripts.sample_parsing_quality import allocate_era_quota


@pytest.mark.unit
def test_allocate_era_quota_sums_to_target_and_respects_pool_sizes() -> None:
    counts = {"2004-2009": 2, "2010-2016": 3, "2017-2021": 2, "2022-2025": 8}

    result = allocate_era_quota(counts, quota=9)

    assert sum(result.values()) == 9
    assert all(0 <= result[era] <= counts[era] for era in counts)


@pytest.mark.unit
def test_allocate_era_quota_never_exceeds_a_thin_eras_population() -> None:
    counts = {"2004-2009": 1, "2010-2016": 20}

    result = allocate_era_quota(counts, quota=5)

    assert sum(result.values()) == 5
    assert result["2004-2009"] <= 1


@pytest.mark.unit
def test_allocate_era_quota_zero_quota_allocates_nothing() -> None:
    counts = {"2004-2009": 2, "2010-2016": 3}

    result = allocate_era_quota(counts, quota=0)

    assert result == {"2004-2009": 0, "2010-2016": 0}
