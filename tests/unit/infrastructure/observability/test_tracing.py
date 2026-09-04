"""Tracing: the switch, the no-op, the prices, and the spans a run produces [M5-07].

The centrepiece is ``test_every_node_appears_as_a_span`` and its neighbours,
which run the **real compiled graph** against a real ``Langfuse`` client whose
span exporter writes to memory instead of to a server. That is what makes the
[M5-07] DoD's "trace every node" and "tag traces with the correlation id"
checkable in CI, with no Langfuse running and no credentials: the assertions are
on spans the production code path actually emitted, not on a mock's call log.

The graph fakes come from ``test_claim_graph`` rather than being restated here
-- one fake LLM and one stub retriever for the whole graph, defined where the
graph itself is tested.

Not covered here: what the [M5-06] correlation id does *outside* the graph
(``tests/integration/test_observability.py``), and anything that needs a live
Langfuse -- the compose ``tracing`` profile is deliberately not a CI dependency.
"""

from dataclasses import replace
from itertools import cycle
from types import SimpleNamespace
from typing import Any, cast

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langfuse import Langfuse
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import SecretStr

from infrastructure.config.enums import LlmProvider
from infrastructure.config.settings import LlmSettings, ObservabilitySettings
from infrastructure.graph.build import build_claim_graph
from infrastructure.observability.tracing import (
    TRACE_NAME,
    LangfuseTracer,
    NullTracer,
    build_tracer,
    model_price_definitions,
    register_model_prices,
)
from tests.unit.infrastructure.graph.test_claim_graph import (
    _STUB_HIT,
    APPROVE,
    _context,
)

# Loopback, port 1: nothing listens, so any SDK call that would go to a server
# is refused immediately rather than hanging. The tests below assert that this
# costs nothing -- an unreachable Langfuse must not be an outage.
DEAD_HOST = "http://127.0.0.1:1"


def _llm_settings(
    *,
    fast_input: float = 0.14,
    fast_output: float = 0.28,
    reasoning_input: float = 1.1154,
    reasoning_output: float = 3.3462,
) -> LlmSettings:
    return LlmSettings(
        LLM_PROVIDER=LlmProvider.OPENAI,
        LLM_API_KEY="test-key",
        LLM_MODEL_FAST="fake-fast-model",
        LLM_MODEL_REASONING="fake-reasoning-model",
        EMBEDDING_MODEL="embed-model",
        RERANKER_MODEL="rerank-model",
        LLM_FAST_INPUT_COST_PER_1M_TOKENS_USD=fast_input,
        LLM_FAST_OUTPUT_COST_PER_1M_TOKENS_USD=fast_output,
        LLM_REASONING_INPUT_COST_PER_1M_TOKENS_USD=reasoning_input,
        LLM_REASONING_OUTPUT_COST_PER_1M_TOKENS_USD=reasoning_output,
        _env_file=None,
    )


def _observability(
    *,
    public_key: str = "pk-lf-test",
    secret_key: str = "sk-lf-test",
    tracing_enabled: bool = True,
) -> ObservabilitySettings:
    return ObservabilitySettings(
        LANGFUSE_PUBLIC_KEY=public_key,
        LANGFUSE_SECRET_KEY=SecretStr(secret_key),
        LANGFUSE_HOST=DEAD_HOST,
        TRACING_ENABLED=tracing_enabled,
        _env_file=None,
    )


def _tracer(public_key: str) -> tuple[LangfuseTracer, InMemorySpanExporter, Langfuse]:
    """A real tracer over a real client that exports to memory.

    ``public_key`` must be unique per test: the SDK keys its client registry on
    it, and ``CallbackHandler`` looks the client up by that key.
    """
    exporter = InMemorySpanExporter()
    client = Langfuse(
        public_key=public_key,
        secret_key="sk-lf-test",
        base_url=DEAD_HOST,
        span_exporter=exporter,
        tracing_enabled=True,
    )
    return LangfuseTracer(client, public_key=public_key), exporter, client


def _run_traced_graph(
    tracer: LangfuseTracer,
    *,
    correlation_id: str = "corr-1",
    assessment_id: str = "a-1",
) -> None:
    """Run the whole graph, checkpoint and resume included, under ``tracer``."""
    context = replace(_context([]), tracer=tracer)
    compiled = build_claim_graph().compile(checkpointer=InMemorySaver())
    config: Any = {
        "configurable": {"thread_id": assessment_id},
        "callbacks": tracer.callbacks(),
    }
    with tracer.assessment_run(
        assessment_id=assessment_id, correlation_id=correlation_id
    ):
        out = compiled.invoke(
            {"claim_id": "c1", "raw_claim_text": "bati o carro"},
            config=config,
            context=context,
        )
        assert "__interrupt__" in out
        compiled.invoke(Command(resume=APPROVE), config=config, context=context)


def _attributes(span: ReadableSpan) -> dict[str, Any]:
    return dict(span.attributes or {})


# ---- the switch --------------------------------------------------------


@pytest.mark.unit
def test_tracing_is_active_only_when_flagged_and_credentialed() -> None:
    assert _observability().tracing_active is True
    assert _observability(tracing_enabled=False).tracing_active is False
    assert _observability(public_key="").tracing_active is False
    assert _observability(secret_key="").tracing_active is False


@pytest.mark.unit
@pytest.mark.parametrize(
    "settings",
    [
        _observability(tracing_enabled=False),
        _observability(public_key=""),
        _observability(secret_key=""),
    ],
    ids=["flag-off", "no-public-key", "no-secret-key"],
)
def test_build_tracer_returns_the_no_op_when_tracing_is_off(
    settings: ObservabilitySettings,
) -> None:
    tracer = build_tracer(observability=settings, llm=_llm_settings())

    assert isinstance(tracer, NullTracer)
    # The point of the switch: no client, so no exporter thread and no keys read.
    assert tracer.callbacks() == []


@pytest.mark.unit
def test_build_tracer_returns_a_langfuse_tracer_when_configured() -> None:
    tracer = build_tracer(
        observability=_observability(),
        llm=_llm_settings(),
        span_exporter=InMemorySpanExporter(),
    )

    # Built against an unreachable host: the price registration and trace-URL
    # lookup both fail, and neither is allowed to stop tracing coming up.
    assert isinstance(tracer, LangfuseTracer)
    tracer.shutdown()


# ---- the no-op ---------------------------------------------------------


@pytest.mark.unit
def test_the_null_tracer_does_nothing_and_raises_nothing() -> None:
    tracer = NullTracer()

    assert tracer.callbacks() == []
    with tracer.assessment_run(assessment_id="a", correlation_id="c"):
        pass
    with tracer.span("retrieval", input={"query": "x"}) as traced:
        traced["n"] = 1
    tracer.shutdown()


@pytest.mark.unit
def test_a_broken_tracer_does_not_break_the_body_it_wraps() -> None:
    """The load-bearing guarantee: observability failure is never a run failure."""

    class _ExplodingClient:
        def __getattr__(self, name: str) -> object:
            raise RuntimeError(f"langfuse is down ({name})")

    tracer = LangfuseTracer(cast(Langfuse, _ExplodingClient()), public_key="pk")

    ran = []
    with tracer.assessment_run(assessment_id="a", correlation_id="c"):
        with tracer.span("retrieval", input={}) as traced:
            traced["n"] = 1
            ran.append("body")

    # Both context managers swallowed the failure and ran their body anyway.
    # (``callbacks()`` is not asserted empty here: the SDK's own handler
    # constructor does not raise for an unknown key, it warns and no-ops.)
    assert ran == ["body"]


# ---- prices ------------------------------------------------------------


@pytest.mark.unit
def test_model_prices_convert_from_per_million_to_per_token() -> None:
    definitions = model_price_definitions(
        _llm_settings(
            fast_input=0.14,
            fast_output=0.28,
            reasoning_input=1.1154,
            reasoning_output=3.3462,
        )
    )

    assert [name for name, _, _ in definitions] == [
        "fake-fast-model",
        "fake-reasoning-model",
    ]
    assert [price for _, price, _ in definitions] == pytest.approx([0.14e-6, 1.1154e-6])
    assert [price for _, _, price in definitions] == pytest.approx([0.28e-6, 3.3462e-6])


@pytest.mark.unit
def test_registered_prices_cover_reasoning_tokens() -> None:
    """Reasoning tokens are a third usage key, and they are most of the bill.

    A flat input/output price pair prices only the ``input`` and ``output``
    keys; a reasoning model reports its thinking under ``output_reasoning``, and
    a measured compatibility call put 3725 of its ~4000 completion tokens there.
    Leaving it unpriced understates the run's cost several-fold.
    """
    captured: list[dict[str, Any]] = []

    class _FakeModels:
        def list(self, *, page: int, limit: int) -> Any:
            return SimpleNamespace(data=[], meta=SimpleNamespace(total_pages=1))

        def upsert(self, model_id: str, **kwargs: Any) -> None:
            captured.append({"id": model_id, **kwargs})

    client = cast(Langfuse, SimpleNamespace(api=SimpleNamespace(models=_FakeModels())))

    register_model_prices(client, _llm_settings())

    assert [call["model_name"] for call in captured] == [
        "fake-fast-model",
        "fake-reasoning-model",
    ]
    prices = captured[1]["pricing_tiers"][0].prices
    assert prices["output_reasoning"] == prices["output"] == pytest.approx(3.3462e-6)
    assert prices["input"] == pytest.approx(1.1154e-6)
    # The API rejects flat prices alongside tiers, so only tiers may be sent.
    assert "input_price" not in captured[1]


# ---- the spans a real run produces -------------------------------------


@pytest.mark.unit
def test_every_node_appears_as_a_span() -> None:
    """[M5-07] DoD: trace every node. Asserted on the real graph's real spans."""
    tracer, exporter, client = _tracer("pk-lf-nodes")
    try:
        _run_traced_graph(tracer)
    finally:
        client.flush()

    names = {span.name for span in exporter.get_finished_spans()}
    assert {
        "intake",
        "retrieval",
        "compatibility",
        "consistency",
        "recommendation",
        "human_review",
    } <= names
    client.shutdown()


@pytest.mark.unit
def test_node_spans_carry_their_input_and_output() -> None:
    tracer, exporter, client = _tracer("pk-lf-io")
    try:
        _run_traced_graph(tracer)
    finally:
        client.flush()

    recommendation = next(
        span for span in exporter.get_finished_spans() if span.name == "recommendation"
    )
    attributes = _attributes(recommendation)
    assert "langfuse.observation.input" in attributes
    assert "recommendation" in str(attributes["langfuse.observation.output"])
    client.shutdown()


@pytest.mark.unit
def test_the_retrieval_span_carries_the_candidate_list_with_scores() -> None:
    """[M5-07] DoD: retrieval as a first-class span with candidates and scores."""
    tracer, exporter, client = _tracer("pk-lf-retrieval")
    try:
        _run_traced_graph(tracer)
    finally:
        client.flush()

    span = next(s for s in exporter.get_finished_spans() if s.name == "retrieval")
    attributes = _attributes(span)
    recorded_input = str(attributes["langfuse.observation.input"])
    output = str(attributes["langfuse.observation.output"])

    assert '"query": "bati o carro"' in recorded_input
    assert '"product_line": "CASCO"' in recorded_input
    assert _STUB_HIT.clause_id in output
    assert f'"score": {_STUB_HIT.score}' in output
    # The gate's reasoning, three fields of which reach no other record.
    assert '"threshold"' in output
    assert '"sufficient": true' in output
    client.shutdown()


@pytest.mark.unit
def test_the_trace_is_tagged_with_the_correlation_id() -> None:
    """[M5-07] DoD: a trace can be reached from a log line."""
    tracer, exporter, client = _tracer("pk-lf-correlation")
    try:
        _run_traced_graph(tracer, correlation_id="corr-m5-07", assessment_id="a-42")
    finally:
        client.flush()

    root = next(s for s in exporter.get_finished_spans() if s.name == TRACE_NAME)
    attributes = _attributes(root)

    assert attributes["langfuse.trace.tags"] == ("corr-m5-07",)
    assert attributes["langfuse.trace.metadata.correlation_id"] == "corr-m5-07"
    # Session id groups the run and its post-decision resume as one assessment.
    assert attributes["session.id"] == "a-42"
    client.shutdown()


@pytest.mark.unit
def test_an_llm_call_becomes_a_generation_with_token_usage() -> None:
    """The tokens half of "inputs, outputs, latency, tokens, cost".

    Driven through a real ``BaseChatModel`` rather than the graph's fake, which
    is a plain runnable and so produces a chain span, not a generation.
    """
    tracer, exporter, client = _tracer("pk-lf-usage")
    reply = AIMessage(
        content="ok",
        usage_metadata={"input_tokens": 11, "output_tokens": 7, "total_tokens": 18},
    )
    model = GenericFakeChatModel(messages=cycle([reply]))

    model.invoke("oi", config={"callbacks": tracer.callbacks()})
    client.flush()

    generations = [
        span
        for span in exporter.get_finished_spans()
        if _attributes(span).get("langfuse.observation.type") == "generation"
    ]
    assert len(generations) == 1
    usage = str(_attributes(generations[0])["langfuse.observation.usage_details"])
    assert '"input": 11' in usage
    assert '"output": 7' in usage
    client.shutdown()
