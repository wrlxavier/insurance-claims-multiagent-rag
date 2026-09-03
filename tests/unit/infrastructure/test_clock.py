"""The SystemClock [M5-04]."""

from datetime import UTC, datetime

import pytest

from infrastructure.clock import SystemClock


@pytest.mark.unit
def test_now_is_timezone_aware_utc_and_current() -> None:
    before = datetime.now(UTC)
    now = SystemClock().now()
    after = datetime.now(UTC)

    assert now.tzinfo is not None
    assert now.utcoffset() == UTC.utcoffset(None)
    assert before <= now <= after
