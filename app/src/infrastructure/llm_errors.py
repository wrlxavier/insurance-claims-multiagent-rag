"""Tell a transient provider fault from a real error [M5-05].

Every LLM call in the graph goes through ``langchain-openai`` -- and so the
``openai`` SDK -- to an OpenAI-compatible gateway (OpenRouter). When a background
assessment run raises, the worker has to decide: reschedule with backoff, or
dead-letter. That decision is *transient vs real*, and this module is the single
place it is made.

Transient -- worth retrying after a wait:

- ``openai.RateLimitError`` (HTTP 429) -- the documented failure mode for this
  project's pinned single-provider routes under sustained load
  (``docs/END_TO_END_EVALUATION.md``);
- ``openai.APITimeoutError`` / ``openai.APIConnectionError`` -- the call never
  got a usable response (the SDK wraps every transport-level fault in one of
  these);
- ``openai.InternalServerError`` and any other ``openai.APIStatusError`` whose
  status is 429 or 5xx -- the upstream is briefly unwell.

Everything else -- a 4xx that is not 429 (a malformed request, auth, a model
that does not exist), a schema-validation error, a bug -- is real: retrying it
just burns the budget.

The raised exception is often wrapped (LangChain re-raises, ``RunAssessment``
re-raises as a typed error), so the whole ``__cause__`` / ``__context__`` chain
is walked.
"""

from __future__ import annotations

import openai

_TRANSIENT_TYPES: tuple[type[BaseException], ...] = (
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.InternalServerError,
)

_TRANSIENT_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


def _is_transient_single(exc: BaseException) -> bool:
    if isinstance(exc, _TRANSIENT_TYPES):
        return True
    if isinstance(exc, openai.APIStatusError):
        return exc.status_code in _TRANSIENT_STATUS_CODES
    return False


def is_transient_llm_error(exc: BaseException) -> bool:
    """Whether ``exc`` (or anything it wraps) is a retryable provider fault."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if _is_transient_single(current):
            return True
        current = current.__cause__ or current.__context__
    return False
