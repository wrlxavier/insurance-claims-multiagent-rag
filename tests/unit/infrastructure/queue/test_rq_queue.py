"""The RqAssessmentQueue adapter -- what it hands to RQ [M5-05]."""

from __future__ import annotations

from typing import Any

import pytest
from rq import Retry

from infrastructure.queue.rq_queue import RqAssessmentQueue


class _FakeRqQueue:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def enqueue(self, f: str, *args: Any, **kwargs: Any) -> None:
        self.calls.append({"f": f, "args": args, "kwargs": kwargs})


def _queue(**overrides: Any) -> tuple[RqAssessmentQueue, _FakeRqQueue]:
    fake = _FakeRqQueue()
    defaults: dict[str, Any] = {
        "job_path": "pkg.mod.run",
        "max_attempts": 3,
        "retry_intervals": [30, 120],
        "job_timeout": 1800,
    }
    defaults.update(overrides)
    return RqAssessmentQueue(fake, **defaults), fake  # type: ignore[arg-type]


@pytest.mark.unit
def test_enqueue_passes_id_job_id_timeout_and_retry() -> None:
    queue, fake = _queue()

    queue.enqueue("assess-42")

    (call,) = fake.calls
    assert call["f"] == "pkg.mod.run"
    assert call["args"] == ("assess-42",)
    assert call["kwargs"]["job_id"] == "assess-42"
    assert call["kwargs"]["job_timeout"] == 1800
    retry = call["kwargs"]["retry"]
    assert isinstance(retry, Retry)
    assert retry.max == 2
    assert retry.intervals == [30, 120]


@pytest.mark.unit
def test_no_retry_object_when_only_one_attempt_is_allowed() -> None:
    queue, fake = _queue(max_attempts=1)

    queue.enqueue("assess-1")

    assert fake.calls[0]["kwargs"]["retry"] is None
