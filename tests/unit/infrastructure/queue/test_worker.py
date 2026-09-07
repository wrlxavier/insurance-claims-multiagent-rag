"""The worker's transient/permanent retry gate [M5-05]."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from application.errors import PermanentAssessmentError, TransientAssessmentError
from infrastructure.queue.worker import stop_retry_on_permanent


@dataclass
class _Job:
    retries_left: int | None = 2


@pytest.mark.unit
def test_a_permanent_failure_has_its_retry_budget_zeroed() -> None:
    job = _Job(retries_left=2)

    result = stop_retry_on_permanent(
        job,  # type: ignore[arg-type]
        PermanentAssessmentError,
        PermanentAssessmentError("bad claim"),
        None,
    )

    assert job.retries_left == 0
    assert result is True  # RQ still runs its default failure handling -> dead-letter


@pytest.mark.unit
def test_a_transient_failure_keeps_its_retry_budget() -> None:
    job = _Job(retries_left=2)

    stop_retry_on_permanent(
        job,  # type: ignore[arg-type]
        TransientAssessmentError,
        TransientAssessmentError("429"),
        None,
    )

    assert job.retries_left == 2
