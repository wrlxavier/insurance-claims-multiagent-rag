#!/usr/bin/env python3
"""Latency and token cost per assessment, over the synthetic claim set [M5-10].

Every M4 eval measured *quality*. This one measures the two numbers a reviewer
asks first -- how long an assessment takes and what it costs -- by running the
**whole compiled graph** (``build_claim_graph()``) over every synthetic claim,
resumed past the [M4-09] human checkpoint with an automatic ``approve``, exactly
as ``scripts/eval_end_to_end.py`` does. Nothing in the graph, the nodes or the
``(state, runtime) -> dict`` convention changes: the measurement is taken from
the outside.

**Latency** comes from the [M5-06] node logger. ``build.py::_instrumented``
already brackets every node run with a ``node.completed`` / ``node.failed`` line
carrying a ``perf_counter``-measured ``duration_ms`` and the run's correlation
id -- its docstring calls that "a head start on [M5-10]". A ``logging.Handler``
on the ``infrastructure.graph.node`` logger captures those lines; a unique
correlation id per claim demultiplexes them. Each node in the [M4-07] parallel
superstep times *itself*, so the per-node numbers are real even inside the
fan-out. End-to-end latency is ``perf_counter`` around the two ``.invoke`` calls
(to the interrupt, then the resume) -- it excludes the human's deliberation
time, there being no human.

**Token cost** comes from a LangChain callback handler on the graph's run
config. It keys LLM calls by ``run_id`` (so the concurrent superstep is safe),
attributes each to its node via ``metadata["langgraph_node"]``, and reads the
**full** ``usage_metadata`` -- including the
``output_token_details["reasoning"]`` breakdown the persisted ``AuditEvent``
``TokenUsage`` does not keep. ``infrastructure.evaluation.token_cost`` prices the
counts from the configured per-1M-token list prices (reasoning tokens at the
output rate, the position [M5-07]'s Langfuse ``output_reasoning`` tier takes).

The report compares the handler's per-node token totals against the
``audit_trail``'s: for the routes measured, LangChain's ``output_tokens`` already
includes the reasoning count, so the audit trail's *totals* are complete and only
the breakdown is missing -- ``docs/PERFORMANCE.md`` has the numbers. Persisting
the reasoning breakdown is a separate follow-up.

Needs a running Postgres with loaded + embedded chunks, the optional ``embed``
uv group, and ``LLM_*`` in ``.env``. Run via ``make eval-performance``
(``LLM_PROVIDER=openai make eval-performance`` locally). Writes
``eval/runs/performance.{md,json}`` + a per-claim
``eval/runs/performance_per_claim.jsonl``; ``--write-snapshot`` also writes the
committed ``eval/performance.json``. The committed analysis lives in
``docs/PERFORMANCE.md``.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import signal
import statistics
import threading
from collections import defaultdict
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from random import Random
from time import perf_counter
from types import FrameType
from typing import Any, cast
from uuid import UUID

from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import LLMResult
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from sqlalchemy.orm import Session, sessionmaker

from infrastructure.config.llm_client_factory import build_chat_model
from infrastructure.config.settings import LlmSettings, get_llm_settings
from infrastructure.database import (
    assert_chunk_table_ready,
    create_engine_from_settings,
    create_session_factory,
)
from infrastructure.evaluation.latency_stats import percentile, summarise_latency
from infrastructure.evaluation.synthetic_claims_schema import SyntheticClaim
from infrastructure.evaluation.token_cost import (
    CallCost,
    ModelRate,
    per_token_rates,
    price_call,
)
from infrastructure.graph import verdict_readout
from infrastructure.graph.build import build_claim_graph
from infrastructure.graph.checkpointer import build_checkpoint_serializer
from infrastructure.graph.context import GraphContext
from infrastructure.graph.state import AuditEvent
from infrastructure.rag.retriever_factory import (
    RetrieverComponents,
    build_graph_retriever,
    load_retriever_components,
)
from scripts.eval_consistency import load_claims
from scripts.eval_end_to_end import (
    _CLAIM_TIMEOUT_SECONDS,
    _RESUME_DECISION,
    _SHUFFLE_SEED,
    build_claim_text,
)
from scripts.eval_retrieval import MANIFEST_PATH, load_document_metadata

_SessionFactory = sessionmaker[Session]

SCHEMA_VERSION = "v1"
OUTPUT_DIR = Path("eval/runs")
DEFAULT_STEM = "performance"
# Committed (eval/runs/ is gitignored, eval/*.json is not).
SNAPSHOT_PATH = Path("eval/performance.json")

# The [M5-06] node logger `build.py::_instrumented` emits `node.start` /
# `node.completed` / `node.failed` on.
NODE_LOGGER_NAME = "infrastructure.graph.node"

# The [M4-07] parallel fan-out members. Token cost is identical whether these
# run concurrently or back to back; latency is not, and that difference is the
# "parallel branch vs sequential" the DoD asks for.
_ASSESSMENT_NODES = ("compatibility", "consistency", "injection_scan")

# The one node on the reasoning model; everything else is the fast model. Used
# to price a call whose reported model name is a provider alias.
_REASONING_NODE = "compatibility"

# Graph order, for a stable per-node table.
_NODE_ORDER = (
    "intake",
    "clarification",
    "clarification_exhausted",
    "retrieval",
    "compatibility",
    "consistency",
    "injection_scan",
    "recommendation",
    "human_review",
)

_FOLLOWUP_NOTE = (
    "Persisting the reasoning-token count in `TokenUsage` / the `audit_event` "
    "table is a separate follow-up (see docs/PERFORMANCE.md)."
)

# Per-request LLM timeout. Generous for a real slow reasoning call (~60-120s
# observed), tight enough that a genuinely stalled upstream fails instead of
# hanging the batch. `max_retries=0` on the client so its retry does not stack
# with the nodes' own `_invoke_with_retry`.
_LLM_TIMEOUT_SECONDS = 200.0


class _ClaimTimeout(BaseException):
    """Raised by the per-claim SIGALRM so a stuck graph run is abandoned.

    A ``BaseException``, not ``Exception``: the nodes' ``_invoke_with_retry``
    catches bare ``Exception`` and would otherwise swallow this into a retry.
    """


def _raise_claim_timeout(signum: int, frame: FrameType | None) -> None:
    raise _ClaimTimeout


@contextmanager
def _claim_deadline(seconds: float) -> Iterator[None]:
    """Arm a SIGALRM for ``seconds``; clear it on the way out.

    Runs in the process main thread (the eval is a script), where ``signal`` is
    available. No worker thread, so a timed-out claim leaves nothing running.
    """
    previous = signal.signal(signal.SIGALRM, _raise_claim_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


# --------------------------------------------------------------------------- #
# Capture: per-node latency from the [M5-06] logger
# --------------------------------------------------------------------------- #


class _NodeLatencyCollector(logging.Handler):
    """Collect ``node.completed`` / ``node.failed`` durations, keyed by run.

    Thread-safe: the [M4-07] fan-out runs ``compatibility`` / ``consistency`` /
    ``injection_scan`` on LangGraph's thread pool, so their log records arrive
    concurrently.
    """

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        # correlation id -> list of (node, duration_ms, failed)
        self._by_run: dict[str, list[tuple[str, float, bool]]] = defaultdict(list)

    def emit(self, record: logging.LogRecord) -> None:
        """Record one node's wall time; ignore every other line."""
        message = record.getMessage()
        if message not in ("node.completed", "node.failed"):
            return
        node = getattr(record, "node", None)
        duration_ms = getattr(record, "duration_ms", None)
        if not isinstance(node, str) or not isinstance(duration_ms, (int, float)):
            return
        run = str(getattr(record, "correlation_id", "-"))
        with self._lock:
            self._by_run[run].append(
                (node, float(duration_ms), message == "node.failed")
            )

    def drain(self, run: str) -> list[tuple[str, float, bool]]:
        """Take and clear the records for one correlation id."""
        with self._lock:
            return self._by_run.pop(run, [])


# --------------------------------------------------------------------------- #
# Capture: per-node token usage from a callback handler
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _LlmCall:
    """One LLM call the graph made, attributed to its node."""

    node: str
    model: str | None
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    cache_read_tokens: int
    failed: bool = False

    @property
    def tier(self) -> str:
        return "reasoning" if self.node == _REASONING_NODE else "fast"


class _NodeCostCallback(BaseCallbackHandler):
    """Attribute every LLM call to its graph node and keep its token usage.

    ``metadata["langgraph_node"]`` is stamped by LangGraph on each node's child
    runnable config and inherited by the bare ``chain.invoke(messages)`` calls
    the nodes make (the same mechanism [M5-07]'s Langfuse handler relies on).
    Keyed by ``run_id`` so the concurrent [M4-07] superstep is unambiguous.
    """

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self._pending: dict[UUID, tuple[str, str | None]] = {}
        self.calls: list[_LlmCall] = []

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Remember which node opened this call."""
        meta = metadata or {}
        node = meta.get("langgraph_node")
        model = meta.get("ls_model_name") or _model_from_serialized(serialized)
        with self._lock:
            self._pending[run_id] = (
                node if isinstance(node, str) else "unknown",
                model if isinstance(model, str) else None,
            )

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Record the call's token usage against its node."""
        with self._lock:
            node, model = self._pending.pop(run_id, ("unknown", None))
        usage = _usage_of(response)
        output_details = usage.get("output_token_details") or {}
        input_details = usage.get("input_token_details") or {}
        with self._lock:
            self.calls.append(
                _LlmCall(
                    node=node,
                    model=model,
                    input_tokens=_as_int(usage.get("input_tokens")),
                    output_tokens=_as_int(usage.get("output_tokens")),
                    reasoning_tokens=_as_int(output_details.get("reasoning")),
                    cache_read_tokens=_as_int(input_details.get("cache_read")),
                )
            )

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """A failed call still cost something upstream; record it, zero tokens."""
        with self._lock:
            node, model = self._pending.pop(run_id, ("unknown", None))
            self.calls.append(_LlmCall(node, model, 0, 0, 0, 0, failed=True))


def _usage_of(response: LLMResult) -> dict[str, Any]:
    """The ``usage_metadata`` dict off an ``LLMResult``, or ``{}``."""
    try:
        message = response.generations[0][0].message  # type: ignore[union-attr]
    except (AttributeError, IndexError):
        return {}
    usage = getattr(message, "usage_metadata", None)
    return usage if isinstance(usage, dict) else {}


def _model_from_serialized(serialized: dict[str, Any] | None) -> str | None:
    kwargs = (serialized or {}).get("kwargs") or {}
    model = kwargs.get("model") or kwargs.get("model_name")
    return model if isinstance(model, str) else None


def _as_int(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


# --------------------------------------------------------------------------- #
# Per-claim measurement
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _LatencySegments:
    """The three latency numbers for one assessment (seconds)."""

    to_interrupt_s: float
    resume_s: float

    @property
    def total_s(self) -> float:
        return self.to_interrupt_s + self.resume_s


@dataclass(frozen=True)
class _ClaimPerf:
    """One claim's whole-graph run, measured."""

    claim_id: str
    cohort: str
    repeat: int
    reached_checkpoint: bool
    completed: bool
    context_sufficient: bool | None
    clarification_rounds: int
    predicted_verdict: str | None
    latency: _LatencySegments
    node_latencies_ms: dict[str, list[float]]
    node_failed: tuple[str, ...]
    calls: tuple[_LlmCall, ...]
    audit_tokens_by_node: dict[str, int]
    error: str | None = None

    def node_contribution_ms(self, node: str) -> float:
        """Total wall time this node cost the assessment (summed over invocations)."""
        return sum(self.node_latencies_ms.get(node, []))

    @property
    def compatibility_ran(self) -> bool:
        return bool(self.node_latencies_ms.get("compatibility"))


@dataclass(frozen=True)
class _GraphDeps:
    """The process-wide pieces every claim's graph run shares."""

    fast_model: BaseChatModel
    reasoning_model: BaseChatModel
    settings: LlmSettings
    session_factory: _SessionFactory
    retriever_components: RetrieverComponents


def _run_graph_measured(
    claim: SyntheticClaim,
    claim_text: str,
    deps: _GraphDeps,
    *,
    correlation_id: str,
) -> tuple[dict[str, Any], bool, _LatencySegments, list[_LlmCall]]:
    """Invoke the compiled graph to the checkpoint, time it, then resume past it.

    Mirrors ``scripts/eval_end_to_end._run_graph``: the project serializer on an
    ``InMemorySaver`` (nothing survives the process; the allowlist keeps state
    models typed), a canned ``approve`` on the interrupt. Opens its **own**
    Postgres session, so a claim abandoned on a hung call (the executor is shut
    down ``wait=False``) does not leave a shared session in use by the next one.
    """
    cost_cb = _NodeCostCallback()
    config: Any = {
        "configurable": {
            "thread_id": f"m5-10-{claim.claim_id}",
            "correlation_id": correlation_id,
        },
        "callbacks": [cost_cb],
    }
    with deps.session_factory() as session:
        assert_chunk_table_ready(session)
        context = GraphContext(
            fast_model=deps.fast_model,
            reasoning_model=deps.reasoning_model,
            retriever=build_graph_retriever(session, deps.retriever_components),
            llm_settings=deps.settings,
            correlation_id=correlation_id,
        )
        compiled = build_claim_graph().compile(
            checkpointer=InMemorySaver(serde=build_checkpoint_serializer())
        )

        started = perf_counter()
        paused = compiled.invoke(
            {"claim_id": claim.claim_id, "raw_claim_text": claim_text},
            config=config,
            context=context,
        )
        to_interrupt_s = perf_counter() - started

        resume_s = 0.0
        reached = "__interrupt__" in paused
        final: dict[str, Any] = cast("dict[str, Any]", paused)
        if reached:
            started = perf_counter()
            final = cast(
                "dict[str, Any]",
                compiled.invoke(
                    Command(resume=_RESUME_DECISION), config=config, context=context
                ),
            )
            resume_s = perf_counter() - started

    return final, reached, _LatencySegments(to_interrupt_s, resume_s), cost_cb.calls


def _score_claim(
    claim: SyntheticClaim,
    cohort: str,
    repeat: int,
    state: dict[str, Any],
    *,
    reached_checkpoint: bool,
    latency: _LatencySegments,
    calls: list[_LlmCall],
    node_records: list[tuple[str, float, bool]],
) -> _ClaimPerf:
    audit_trail = cast("list[AuditEvent]", state.get("audit_trail") or [])
    node_latencies: dict[str, list[float]] = defaultdict(list)
    failed: list[str] = []
    for node, duration_ms, is_failed in node_records:
        node_latencies[node].append(duration_ms)
        if is_failed:
            failed.append(node)

    audit_tokens: dict[str, int] = defaultdict(int)
    for event in audit_trail:
        if event.token_usage is not None:
            audit_tokens[event.node] += event.token_usage.total_tokens

    verdict = verdict_readout.effective_verdict(audit_trail)
    return _ClaimPerf(
        claim_id=claim.claim_id,
        cohort=cohort,
        repeat=repeat,
        reached_checkpoint=reached_checkpoint,
        completed=state.get("recommendation") is not None,
        context_sufficient=cast("bool | None", state.get("context_sufficient")),
        clarification_rounds=int(state.get("clarification_rounds", 0) or 0),
        predicted_verdict=verdict.value if verdict is not None else None,
        latency=latency,
        node_latencies_ms=dict(node_latencies),
        node_failed=tuple(failed),
        calls=tuple(calls),
        audit_tokens_by_node=dict(audit_tokens),
    )


def _errored_claim(
    claim: SyntheticClaim, cohort: str, repeat: int, error: str
) -> _ClaimPerf:
    return _ClaimPerf(
        claim_id=claim.claim_id,
        cohort=cohort,
        repeat=repeat,
        reached_checkpoint=False,
        completed=False,
        context_sufficient=None,
        clarification_rounds=0,
        predicted_verdict=None,
        latency=_LatencySegments(0.0, 0.0),
        node_latencies_ms={},
        node_failed=(),
        calls=(),
        audit_tokens_by_node={},
        error=error,
    )


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #


def run_performance_eval(
    *, limit: int | None = None, repeats: int = 1
) -> PerformanceResult:
    """Run the whole graph over every synthetic claim and measure latency + cost."""
    settings = get_llm_settings()
    reasoning_model = build_chat_model(
        settings,
        settings.llm_model_reasoning,
        provider_order=settings.llm_reasoning_provider_order,
        allow_fallbacks=settings.llm_reasoning_allow_fallbacks,
        timeout=_LLM_TIMEOUT_SECONDS,
        max_retries=0,
    )
    fast_model = build_chat_model(
        settings,
        settings.llm_model_fast,
        provider_order=settings.llm_fast_provider_order,
        allow_fallbacks=settings.llm_fast_allow_fallbacks,
        timeout=_LLM_TIMEOUT_SECONDS,
        max_retries=0,
    )
    document_meta = load_document_metadata(MANIFEST_PATH)
    claims = load_claims()
    if limit is not None:
        Random(_SHUFFLE_SEED).shuffle(claims)
        claims = claims[:limit]
    retriever_components = load_retriever_components()

    collector = _NodeLatencyCollector()
    node_logger = logging.getLogger(NODE_LOGGER_NAME)
    node_logger.addHandler(collector)
    # `make migrate`'s Alembic env.py can flip this logger off in a shared test
    # process; the same guard `tests/integration/test_observability.py` uses.
    prior_disabled = node_logger.disabled
    node_logger.disabled = False
    prior_level = node_logger.level
    if not node_logger.isEnabledFor(logging.INFO):
        node_logger.setLevel(logging.INFO)

    engine = create_engine_from_settings()
    deps = _GraphDeps(
        fast_model=fast_model,
        reasoning_model=reasoning_model,
        settings=settings,
        session_factory=create_session_factory(engine=engine),
        retriever_components=retriever_components,
    )
    rows: list[_ClaimPerf] = []
    errors: list[str] = []
    try:
        for repeat in range(1, repeats + 1):
            for cohort, claim in claims:
                correlation_id = f"m5-10-{claim.claim_id}-r{repeat}"
                claim_text = build_claim_text(
                    claim, document_meta[claim.document_id], policy_header=True
                )
                try:
                    with _claim_deadline(_CLAIM_TIMEOUT_SECONDS):
                        state, reached, latency, calls = _run_graph_measured(
                            claim, claim_text, deps, correlation_id=correlation_id
                        )
                except (Exception, _ClaimTimeout) as exc:  # noqa: BLE001 - recorded
                    collector.drain(correlation_id)
                    errors.append(claim.claim_id)
                    reason = (
                        f"timed out after {_CLAIM_TIMEOUT_SECONDS:.0f}s"
                        if isinstance(exc, _ClaimTimeout)
                        else repr(exc)
                    )
                    rows.append(_errored_claim(claim, cohort, repeat, reason))
                    print(f"{claim.claim_id:<28} r{repeat} ERROR {reason}", flush=True)
                    continue
                node_records = collector.drain(correlation_id)
                row = _score_claim(
                    claim,
                    cohort,
                    repeat,
                    state,
                    reached_checkpoint=reached,
                    latency=latency,
                    calls=calls,
                    node_records=node_records,
                )
                rows.append(row)
                tokens = sum(c.input_tokens + c.output_tokens for c in row.calls)
                print(
                    f"{row.claim_id:<28} r{repeat} "
                    f"{row.latency.total_s:>6.1f}s {tokens:>7d} tk "
                    f"{len(row.calls)} calls "
                    f"{'OK' if row.completed else 'INCOMPLETE'}",
                    flush=True,
                )
    finally:
        node_logger.removeHandler(collector)
        node_logger.disabled = prior_disabled
        node_logger.setLevel(prior_level)
        engine.dispose()

    meta = _build_meta(settings=settings, claims=claims, repeats=repeats, limit=limit)
    return _summarise(rows, errors, settings=settings, meta=meta)


def _build_meta(
    *,
    settings: LlmSettings,
    claims: Sequence[tuple[str, SyntheticClaim]],
    repeats: int,
    limit: int | None,
) -> dict[str, Any]:
    from collections import Counter

    counts = Counter(cohort for cohort, _ in claims)
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "platform": platform.platform(),
        "reasoning_model": settings.llm_model_reasoning,
        "fast_model": settings.llm_model_fast,
        "reasoning_provider_order": settings.llm_reasoning_provider_order,
        "fast_provider_order": settings.llm_fast_provider_order,
        "reasoning_price_per_1m_usd": [
            settings.llm_reasoning_input_cost_per_1m_tokens_usd,
            settings.llm_reasoning_output_cost_per_1m_tokens_usd,
        ],
        "fast_price_per_1m_usd": [
            settings.llm_fast_input_cost_per_1m_tokens_usd,
            settings.llm_fast_output_cost_per_1m_tokens_usd,
        ],
        "claim_count": len(claims),
        "cohort_counts": dict(sorted(counts.items())),
        "repeats": repeats,
        "limit": limit,
        "claim_timeout_seconds": _CLAIM_TIMEOUT_SECONDS,
        "method_note": (
            "The whole compiled graph per claim, policy-header arm, resumed past "
            "the [M4-09] checkpoint with an automatic approve. End-to-end latency "
            "is perf_counter around the two .invoke calls and excludes human "
            "deliberation. Per-node latency is the [M5-06] node logger's "
            "duration_ms. Token usage is a callback handler reading the full "
            "usage_metadata (incl. output_token_details.reasoning), priced from "
            "the configured per-1M-token list prices. Prices are OpenRouter "
            "route-specific."
        ),
    }


# --------------------------------------------------------------------------- #
# Aggregate
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _NodeLatency:
    node: str
    invocations: int
    calls_per_assessment: float
    p50_ms: float
    p95_ms: float
    mean_ms: float
    mean_ms_per_assessment: float


@dataclass(frozen=True)
class _NodeCost:
    node: str
    n_calls: int
    mean_input_tokens: float
    mean_output_tokens: float
    mean_reasoning_tokens: float
    mean_usd: float
    p95_usd: float
    audit_trail_tokens: int
    handler_tokens: int


@dataclass(frozen=True)
class _ParallelComparison:
    n_claims: int
    sequential_mean_ms: float
    parallel_mean_ms: float
    saving_p50_ms: float
    saving_p95_ms: float
    saving_mean_ms: float
    saving_fraction_of_stage: float
    saving_fraction_of_end_to_end: float
    token_cost_identical_usd: float


@dataclass(frozen=True)
class PerformanceResult:
    """Everything ``make eval-performance`` produces, for the report + the test."""

    meta: dict[str, Any]
    n_claims: int
    n_scored: int
    error_claim_ids: list[str]
    end_to_end_latency: dict[str, float]
    latency_split: dict[str, float]
    per_node_latency: list[_NodeLatency]
    cost_per_assessment: dict[str, float]
    per_node_cost: list[_NodeCost]
    parallel: _ParallelComparison | None
    rows: list[_ClaimPerf] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        """The JSON-serialisable view written to ``eval/runs/performance.json``."""
        return {
            "schema_version": SCHEMA_VERSION,
            "meta": self.meta,
            "n_claims": self.n_claims,
            "n_scored": self.n_scored,
            "error_claim_ids": self.error_claim_ids,
            "end_to_end_latency_seconds": self.end_to_end_latency,
            "latency_split_seconds": self.latency_split,
            "per_node_latency_ms": [vars(n) for n in self.per_node_latency],
            "cost_per_assessment_usd": self.cost_per_assessment,
            "per_node_cost": [vars(n) for n in self.per_node_cost],
            "parallel_vs_sequential": (
                vars(self.parallel) if self.parallel is not None else None
            ),
        }


def _call_cost(
    call: _LlmCall,
    *,
    rates: dict[str, ModelRate],
    fast_model: str,
    reasoning_model: str,
) -> CallCost:
    return price_call(
        model=call.model,
        tier=call.tier,
        input_tokens=call.input_tokens,
        output_tokens=call.output_tokens,
        reasoning_tokens=call.reasoning_tokens,
        rates=rates,
        fast_model=fast_model,
        reasoning_model=reasoning_model,
    )


def _summarise(
    rows: list[_ClaimPerf],
    errors: list[str],
    *,
    settings: LlmSettings,
    meta: dict[str, Any],
) -> PerformanceResult:
    scored = [r for r in rows if r.error is None]
    rates = per_token_rates(settings)
    fast_model = settings.llm_model_fast
    reasoning_model = settings.llm_model_reasoning

    def claim_cost(row: _ClaimPerf) -> float:
        return sum(
            _call_cost(
                c, rates=rates, fast_model=fast_model, reasoning_model=reasoning_model
            ).total_usd
            for c in row.calls
        )

    end_to_end = summarise_latency([r.latency.total_s for r in scored], digits=2)
    latency_split = {
        "mean_to_interrupt_s": _mean([r.latency.to_interrupt_s for r in scored]),
        "mean_resume_s": _mean([r.latency.resume_s for r in scored]),
        "min_total_s": min((r.latency.total_s for r in scored), default=0.0),
        "max_total_s": max((r.latency.total_s for r in scored), default=0.0),
    }

    # --- per-node latency ------------------------------------------------- #
    per_node_latency: list[_NodeLatency] = []
    n_assessments = len(scored) or 1
    seen_nodes = {n for r in scored for n in r.node_latencies_ms}
    for node in [*_NODE_ORDER, *sorted(seen_nodes - set(_NODE_ORDER))]:
        samples = [ms for r in scored for ms in r.node_latencies_ms.get(node, [])]
        if not samples:
            continue
        stats = summarise_latency(samples, digits=1)
        contributions = [r.node_contribution_ms(node) for r in scored]
        per_node_latency.append(
            _NodeLatency(
                node=node,
                invocations=len(samples),
                calls_per_assessment=round(len(samples) / n_assessments, 2),
                p50_ms=stats["p50"],
                p95_ms=stats["p95"],
                mean_ms=stats["mean"],
                mean_ms_per_assessment=round(_mean(contributions), 1),
            )
        )

    # --- token cost ----------------------------------------------------- #
    per_claim_cost = [claim_cost(r) for r in scored]
    cost_per_assessment = {
        "mean_usd": _round(_mean(per_claim_cost), 6),
        "p95_usd": _round(
            percentile(per_claim_cost, 0.95) if per_claim_cost else 0.0, 6
        ),
        "total_run_usd": _round(sum(per_claim_cost), 4),
        "total_input_tokens": sum(c.input_tokens for r in scored for c in r.calls),
        "total_output_tokens": sum(c.output_tokens for r in scored for c in r.calls),
        "total_reasoning_tokens": sum(
            c.reasoning_tokens for r in scored for c in r.calls
        ),
    }

    per_node_cost = _per_node_cost(
        scored,
        rates=rates,
        fast_model=fast_model,
        reasoning_model=reasoning_model,
    )

    parallel = _parallel_comparison(
        scored, rates=rates, fast_model=fast_model, reasoning_model=reasoning_model
    )

    return PerformanceResult(
        meta=meta,
        n_claims=len(rows),
        n_scored=len(scored),
        error_claim_ids=errors,
        end_to_end_latency=end_to_end,
        latency_split=latency_split,
        per_node_latency=per_node_latency,
        cost_per_assessment=cost_per_assessment,
        per_node_cost=per_node_cost,
        parallel=parallel,
        rows=rows,
    )


def _per_node_cost(
    scored: list[_ClaimPerf],
    *,
    rates: dict[str, ModelRate],
    fast_model: str,
    reasoning_model: str,
) -> list[_NodeCost]:
    nodes = {c.node for r in scored for c in r.calls}
    out: list[_NodeCost] = []
    for node in [*_NODE_ORDER, *sorted(nodes - set(_NODE_ORDER))]:
        node_calls = [c for r in scored for c in r.calls if c.node == node]
        if not node_calls:
            continue
        per_claim_usd = [
            sum(
                _call_cost(
                    c,
                    rates=rates,
                    fast_model=fast_model,
                    reasoning_model=reasoning_model,
                ).total_usd
                for c in r.calls
                if c.node == node
            )
            for r in scored
        ]
        n = len(node_calls)
        out.append(
            _NodeCost(
                node=node,
                n_calls=n,
                mean_input_tokens=round(sum(c.input_tokens for c in node_calls) / n, 1),
                mean_output_tokens=round(
                    sum(c.output_tokens for c in node_calls) / n, 1
                ),
                mean_reasoning_tokens=round(
                    sum(c.reasoning_tokens for c in node_calls) / n, 1
                ),
                mean_usd=_round(_mean(per_claim_usd), 6),
                p95_usd=_round(
                    percentile(per_claim_usd, 0.95) if per_claim_usd else 0.0, 6
                ),
                handler_tokens=sum(
                    c.input_tokens + c.output_tokens for c in node_calls
                ),
                audit_trail_tokens=sum(
                    r.audit_tokens_by_node.get(node, 0) for r in scored
                ),
            )
        )
    return out


def _parallel_comparison(
    scored: list[_ClaimPerf],
    *,
    rates: dict[str, ModelRate],
    fast_model: str,
    reasoning_model: str,
) -> _ParallelComparison | None:
    on_path = [r for r in scored if r.compatibility_ran]
    if not on_path:
        return None
    seq_ms: list[float] = []
    par_ms: list[float] = []
    savings: list[float] = []
    saving_fraction_e2e: list[float] = []
    for r in on_path:
        parts = [r.node_contribution_ms(n) for n in _ASSESSMENT_NODES]
        t_seq = sum(parts)
        t_par = max(parts)
        seq_ms.append(t_seq)
        par_ms.append(t_par)
        savings.append(t_seq - t_par)
        if r.latency.total_s > 0:
            saving_fraction_e2e.append((t_seq - t_par) / 1000.0 / r.latency.total_s)

    identical_cost = sum(
        _call_cost(
            c, rates=rates, fast_model=fast_model, reasoning_model=reasoning_model
        ).total_usd
        for r in on_path
        for c in r.calls
        if c.node in _ASSESSMENT_NODES
    )
    return _ParallelComparison(
        n_claims=len(on_path),
        sequential_mean_ms=round(_mean(seq_ms), 1),
        parallel_mean_ms=round(_mean(par_ms), 1),
        saving_p50_ms=round(percentile(savings, 0.50), 1),
        saving_p95_ms=round(percentile(savings, 0.95), 1),
        saving_mean_ms=round(_mean(savings), 1),
        saving_fraction_of_stage=round(
            sum(savings) / sum(seq_ms) if sum(seq_ms) else 0.0, 4
        ),
        saving_fraction_of_end_to_end=round(_mean(saving_fraction_e2e), 4),
        token_cost_identical_usd=_round(identical_cost, 4),
    )


def _mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _round(value: float, digits: int) -> float:
    return round(value, digits)


# --------------------------------------------------------------------------- #
# Render
# --------------------------------------------------------------------------- #


def _fmt_usd(value: float) -> str:
    return f"${value:.6f}" if value < 0.01 else f"${value:.4f}"


def _table(header: Sequence[str], rows: Sequence[Sequence[object]]) -> list[str]:
    """A GitHub-flavoured Markdown table; first column left, the rest right."""
    align = ["---", *["---:" for _ in header[1:]]]
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join(align) + " |"]
    out += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return out


def render_markdown(result: PerformanceResult) -> str:
    """Render the run as Markdown; the numbers are copied into docs/PERFORMANCE.md."""
    meta = result.meta
    e2e = result.end_to_end_latency
    split = result.latency_split
    cost = result.cost_per_assessment
    reasoning_price = "/".join(str(p) for p in meta["reasoning_price_per_1m_usd"])
    fast_price = "/".join(str(p) for p in meta["fast_price_per_1m_usd"])
    lines = [
        "# Performance: latency and cost per assessment -- measurement ([M5-10])",
        "",
        "Generated by `scripts/eval_performance.py` (`make eval-performance`): the "
        f"whole compiled graph over the synthetic claim set ({meta['claim_count']} "
        f"claims, {meta['repeats']} repeat(s)), reasoning model "
        f"`{meta['reasoning_model']}`, fast model `{meta['fast_model']}`. "
        "Regenerable; committed analysis in `docs/PERFORMANCE.md`.",
        "",
        f"- Generated (UTC): {meta['generated_at_utc']}",
        f"- Platform: {meta['platform']}",
        f"- Reasoning model: `{meta['reasoning_model']}` "
        f"(order `{meta['reasoning_provider_order']}`, ${reasoning_price} per 1M)",
        f"- Fast model: `{meta['fast_model']}` "
        f"(order `{meta['fast_provider_order']}`, ${fast_price} per 1M)",
        f"- Scored: {result.n_scored} of {result.n_claims}  "
        f"(errors: {result.error_claim_ids or 'none'})",
        f"- Method: {meta['method_note']}",
        "",
        "## End-to-end latency",
        "",
    ]
    lines += _table(
        ["metric", "seconds"],
        [
            ["p50", f"{e2e['p50']:.2f}"],
            ["p95", f"{e2e['p95']:.2f}"],
            ["mean", f"{e2e['mean']:.2f}"],
            ["min", f"{split['min_total_s']:.2f}"],
            ["max", f"{split['max_total_s']:.2f}"],
        ],
    )
    lines += [
        "",
        f"Split: mean {split['mean_to_interrupt_s']:.2f} s to the checkpoint + "
        f"{split['mean_resume_s']:.2f} s to resume past it. Human deliberation is "
        "not included (the checkpoint is resumed mechanically).",
        "",
        "## Per-node latency",
        "",
    ]
    lines += _table(
        ["node", "invocations", "calls/asmt", "p50 ms", "p95 ms", "mean ms", "ms/asmt"],
        [
            [
                n.node,
                n.invocations,
                n.calls_per_assessment,
                n.p50_ms,
                n.p95_ms,
                n.mean_ms,
                n.mean_ms_per_assessment,
            ]
            for n in result.per_node_latency
        ],
    )
    lines += [
        "",
        "`compatibility`, `consistency` and `injection_scan` run in one [M4-07] "
        "superstep; each times itself, so these rows are real, not additive.",
        "",
        "## Token cost per assessment",
        "",
        f"**Mean {_fmt_usd(cost['mean_usd'])} / assessment, p95 "
        f"{_fmt_usd(cost['p95_usd'])}.** Whole run: {_fmt_usd(cost['total_run_usd'])} "
        f"over {cost['total_input_tokens']:,} input + "
        f"{cost['total_output_tokens']:,} output tokens "
        f"({cost['total_reasoning_tokens']:,} of the output was reasoning).",
        "",
    ]
    lines += _table(
        ["node", "calls", "in tok", "out tok", "reasoning tok", "mean $", "p95 $"],
        [
            [
                c.node,
                c.n_calls,
                c.mean_input_tokens,
                c.mean_output_tokens,
                c.mean_reasoning_tokens,
                _fmt_usd(c.mean_usd),
                _fmt_usd(c.p95_usd),
            ]
            for c in result.per_node_cost
        ],
    )
    lines += [
        "",
        "Reasoning tokens are priced at the output rate and are already inside "
        "`out tok` -- the column is a memo, not an addend.",
        "",
        "## Parallel branch vs sequential",
        "",
    ]
    if result.parallel is None:
        lines.append("_No claim reached the parallel assessment branch._")
    else:
        p = result.parallel
        identical = _fmt_usd(p.token_cost_identical_usd)
        lines += [
            f"- **Token cost is identical** either way: {identical} of "
            "assessment-branch LLM calls, whether they run concurrently or back "
            "to back -- parallelism changes scheduling, not the call set.",
            f"- **Latency saving** ({p.n_claims} claims on the branch): mean "
            f"{p.saving_mean_ms / 1000:.1f} s (p50 {p.saving_p50_ms / 1000:.1f} s, "
            f"p95 {p.saving_p95_ms / 1000:.1f} s) -- "
            f"{p.saving_fraction_of_stage:.1%} of the assessment stage, "
            f"{p.saving_fraction_of_end_to_end:.1%} of end-to-end latency.",
            f"- Sequential mean {p.sequential_mean_ms / 1000:.1f} s vs parallel "
            f"mean {p.parallel_mean_ms / 1000:.1f} s.",
            "",
            "See `docs/PARALLEL_ASSESSMENT.md` ([M4-07]) for the isolated fan-out "
            "measurement this confirms at claim-set scale.",
        ]
    lines += [
        "",
        "## One-off corpus indexing cost",
        "",
        "_Reported separately -- never folded into the per-assessment number. See "
        "`docs/PERFORMANCE.md` for the table (embedding: $0.00 / ~41 min CPU; "
        "vision boundary escalation: ~$2.85; clause classification: estimate)._",
        "",
        "## Reasoning-token measurement note",
        "",
        "The persisted `audit_trail` `TokenUsage` captures `input`/`output`/"
        "`total` only -- not the `output_token_details.reasoning` breakdown. "
        "These numbers come from a callback handler reading the full "
        "`usage_metadata`; the table below shows whether the audit trail's "
        "totals are complete (handler == audit-trail) or short (see "
        "`docs/PERFORMANCE.md`).",
        "",
    ]
    lines += _table(
        ["node", "handler tokens", "audit-trail tokens"],
        [
            [c.node, f"{c.handler_tokens:,}", f"{c.audit_trail_tokens:,}"]
            for c in result.per_node_cost
        ],
    )
    lines += [
        "",
        f"_{_FOLLOWUP_NOTE}_",
        "",
        "## Findings",
        "",
        "_To be written from the first full run._",
        "",
    ]
    return "\n".join(lines) + "\n"


def snapshot_payload(result: PerformanceResult) -> dict[str, Any]:
    """The committed `eval/performance.json` -- aggregates only, no per-claim rows."""
    payload = result.to_json()
    payload["provenance"] = {
        "generated_by": "scripts/eval_performance.py",
        "command": "make eval-performance",
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": result.meta["generated_at_utc"],
        "reasoning_model": result.meta["reasoning_model"],
        "fast_model": result.meta["fast_model"],
        "claim_count": result.meta["claim_count"],
    }
    return payload


def _per_claim_row(r: _ClaimPerf) -> dict[str, Any]:
    return {
        "claim_id": r.claim_id,
        "cohort": r.cohort,
        "repeat": r.repeat,
        "reached_checkpoint": r.reached_checkpoint,
        "completed": r.completed,
        "context_sufficient": r.context_sufficient,
        "clarification_rounds": r.clarification_rounds,
        "predicted_verdict": r.predicted_verdict,
        "latency_total_s": round(r.latency.total_s, 3),
        "latency_to_interrupt_s": round(r.latency.to_interrupt_s, 3),
        "latency_resume_s": round(r.latency.resume_s, 3),
        "node_latencies_ms": {
            k: [round(v, 2) for v in vs] for k, vs in r.node_latencies_ms.items()
        },
        "node_failed": list(r.node_failed),
        "calls": [
            {
                "node": c.node,
                "model": c.model,
                "input_tokens": c.input_tokens,
                "output_tokens": c.output_tokens,
                "reasoning_tokens": c.reasoning_tokens,
                "cache_read_tokens": c.cache_read_tokens,
                "failed": c.failed,
            }
            for c in r.calls
        ],
        "audit_tokens_by_node": r.audit_tokens_by_node,
        "error": r.error,
    }


def main() -> None:
    """Run the measurement and write ``eval/runs/performance.{md,json}``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit", type=int, default=None, help="cap the number of claims (smoke runs)"
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="re-run the whole claim set N times for run-to-run variance",
    )
    parser.add_argument(
        "--out-stem", default=DEFAULT_STEM, help="basename under eval/runs/"
    )
    parser.add_argument(
        "--write-snapshot",
        action="store_true",
        help=f"also write the committed {SNAPSHOT_PATH}",
    )
    args = parser.parse_args()

    result = run_performance_eval(limit=args.limit, repeats=args.repeats)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / f"{args.out_stem}.json"
    md_path = OUTPUT_DIR / f"{args.out_stem}.md"
    per_claim_path = OUTPUT_DIR / f"{args.out_stem}_per_claim.jsonl"
    json_path.write_text(
        json.dumps(result.to_json(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(render_markdown(result), encoding="utf-8")
    with per_claim_path.open("w", encoding="utf-8") as handle:
        for row in result.rows:
            handle.write(json.dumps(_per_claim_row(row), ensure_ascii=False) + "\n")
    written = [str(json_path), str(md_path), str(per_claim_path)]
    if args.write_snapshot:
        SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
        SNAPSHOT_PATH.write_text(
            json.dumps(snapshot_payload(result), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        written.append(str(SNAPSHOT_PATH))

    e2e = result.end_to_end_latency
    cost = result.cost_per_assessment
    print("")
    print(
        f"latency p50 {e2e['p50']:.1f}s / p95 {e2e['p95']:.1f}s | "
        f"cost mean {_fmt_usd(cost['mean_usd'])} / p95 {_fmt_usd(cost['p95_usd'])} | "
        f"scored {result.n_scored}/{result.n_claims} | "
        f"errors {result.error_claim_ids or 'none'}"
    )
    print("Wrote " + ", ".join(written))


if __name__ == "__main__":
    main()
