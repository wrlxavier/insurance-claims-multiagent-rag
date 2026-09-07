"""The correlation-id context variable -- [M5-06]."""

from __future__ import annotations

import pytest

from infrastructure.observability.correlation import (
    bind_correlation_id,
    generate_correlation_id,
    get_correlation_id,
)


@pytest.mark.unit
def test_unset_by_default() -> None:
    assert get_correlation_id() is None


@pytest.mark.unit
def test_bind_sets_and_resets() -> None:
    with bind_correlation_id("abc"):
        assert get_correlation_id() == "abc"
        with bind_correlation_id("def"):
            assert get_correlation_id() == "def"
        assert get_correlation_id() == "abc"
    assert get_correlation_id() is None


@pytest.mark.unit
def test_generate_is_non_empty_and_unique() -> None:
    first, second = generate_correlation_id(), generate_correlation_id()
    assert first and second and first != second
