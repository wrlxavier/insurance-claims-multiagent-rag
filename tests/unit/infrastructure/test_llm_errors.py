"""The transient-vs-real classifier [M5-05]."""

from __future__ import annotations

import httpx2
import openai
import pytest

from infrastructure.llm_errors import is_transient_llm_error

_REQUEST = httpx2.Request("POST", "https://example/api")


def _api_status_error(status: int) -> openai.APIStatusError:
    response = httpx2.Response(status, request=_REQUEST)
    return openai.APIStatusError("boom", response=response, body=None)


@pytest.mark.unit
@pytest.mark.parametrize(
    "exc",
    [
        openai.APITimeoutError(request=_REQUEST),
        openai.APIConnectionError(message="refused", request=_REQUEST),
        _api_status_error(429),
        _api_status_error(503),
    ],
)
def test_transient_faults_are_recognised(exc: BaseException) -> None:
    assert is_transient_llm_error(exc) is True


@pytest.mark.unit
@pytest.mark.parametrize(
    "exc",
    [
        ValueError("schema mismatch"),
        _api_status_error(400),
        _api_status_error(404),
        RuntimeError("bug"),
    ],
)
def test_real_errors_are_not_transient(exc: BaseException) -> None:
    assert is_transient_llm_error(exc) is False


@pytest.mark.unit
def test_a_wrapped_transient_cause_is_found_through_the_chain() -> None:
    try:
        try:
            raise _api_status_error(429)
        except openai.APIStatusError as inner:
            raise RuntimeError("node failed") from inner
    except RuntimeError as outer:
        assert is_transient_llm_error(outer) is True


@pytest.mark.unit
def test_a_cycle_in_the_cause_chain_terminates() -> None:
    first = RuntimeError("a")
    second = RuntimeError("b")
    first.__cause__ = second
    second.__cause__ = first
    assert is_transient_llm_error(first) is False
