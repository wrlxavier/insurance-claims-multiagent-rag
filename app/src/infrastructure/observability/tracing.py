"""Langfuse tracing for the assessment graph -- [M5-07].

[M5-06] made a run *loggable*: JSON lines, one correlation id from the HTTP
request through the queue into every node. Logs answer what happened. They do
not answer which node was wrong, because the thing you need then is the node's
input beside its output beside the clauses it was reasoning over, and that is a
tree, not a sequence of lines.

The whole design is one sentence: **one Langfuse client, installed as a LangChain
callback handler on the graph config**. LangGraph nodes and the
``chain.invoke(messages)`` calls inside them are LangChain runnables, so a
handler on the run's config sees every one of them and produces a span per node
and a generation per LLM call -- with the prompt, the completion, the model and
the token usage -- for no change to ``build.py``, to any node, or to the
``(state, runtime) -> dict`` convention its AST test enforces. Two things the
callbacks cannot see are added by hand:

- **the retrieval span**, because retrieval makes no LLM call: the node opens it
  through [infrastructure.graph.context.TracePort], whose default is a no-op;
- **cost**, because Langfuse prices a generation from a model definition, and it
  has never heard of an OpenRouter model id. ``register_model_prices`` upserts
  one definition per model from the configured per-1M-token prices.

Everything here is off unless ``ObservabilitySettings.tracing_active`` -- the
flag *and* both keys. Off, ``build_tracer`` returns [NullTracer] and no Langfuse
object is ever constructed. On, every call into the SDK is wrapped: a run must
not fail because its observability did, so the guards below log and carry on
rather than raising into a claim assessment.

Import this module directly (``from infrastructure.observability.tracing import
...``); the package ``__init__`` deliberately re-exports nothing.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator, Mapping, MutableMapping
from contextlib import (
    AbstractContextManager,
    ExitStack,
    contextmanager,
    nullcontext,
    suppress,
)
from typing import TYPE_CHECKING, Protocol

from langfuse import Langfuse, propagate_attributes
from langfuse.api.commons.types.pricing_tier_input import PricingTierInput
from langfuse.langchain import CallbackHandler

if TYPE_CHECKING:
    from langchain_core.callbacks.base import BaseCallbackHandler
    from opentelemetry.sdk.trace.export import SpanExporter

    from infrastructure.config.settings import LlmSettings, ObservabilitySettings

logger = logging.getLogger("infrastructure.observability.tracing")

# The name every assessment trace carries, so the Langfuse trace list is a list
# of assessments rather than a list of whatever LangGraph called its root run.
TRACE_NAME = "claim-assessment"

# Langfuse prices a generation per *unit*; the settings are per 1M tokens.
_TOKENS_PER_PRICE_UNIT = 1_000_000

# The model-definitions endpoint's page size cap.
_MODEL_PAGE_SIZE = 100

# Stand-in trace id used once, at startup, to learn the shape of a trace URL --
# see `_resolve_trace_url_template`.
_TRACE_ID_PLACEHOLDER = "0" * 32


class RunTracer(Protocol):
    """One assessment run's tracing, as the orchestrator and nodes need it.

    Two audiences in one object because they are one decision: ``callbacks`` and
    ``assessment_run`` are the orchestrator's (install the handler, open the
    root span, flush), ``span`` is [infrastructure.graph.context.TracePort],
    which is what a node sees. [NullTracer] and [LangfuseTracer] both satisfy
    it, and the orchestrator's default is the null one -- so every existing
    test and eval script builds an untraced orchestrator by doing nothing.
    """

    def callbacks(self) -> list[BaseCallbackHandler]:
        """Handlers to put on the graph's ``config["callbacks"]``."""
        ...

    def assessment_run(
        self, *, assessment_id: str, correlation_id: str
    ) -> AbstractContextManager[None]:
        """Wrap one graph invocation: root span, trace attributes, flush."""
        ...

    def span(
        self,
        name: str,
        *,
        input: Mapping[str, object],
        metadata: Mapping[str, object] | None = None,
    ) -> AbstractContextManager[MutableMapping[str, object]]:
        """Open a child span; fill the yielded mapping to set its output."""
        ...

    def shutdown(self) -> None:
        """Flush and stop the exporter. Called once, on process shutdown."""
        ...


class NullTracer:
    """Tracing switched off: every method is a no-op that allocates nothing.

    Not an error path. An unconfigured clone, a unit test and a deployment with
    ``TRACING_ENABLED=false`` all run the identical code path through the graph;
    the only difference is that these calls do nothing.
    """

    def callbacks(self) -> list[BaseCallbackHandler]:
        """No handlers, so LangGraph's config carries no callbacks at all."""
        return []

    def assessment_run(
        self, *, assessment_id: str, correlation_id: str
    ) -> AbstractContextManager[None]:
        """Run the assessment untraced."""
        return nullcontext()

    def span(
        self,
        name: str,
        *,
        input: Mapping[str, object],
        metadata: Mapping[str, object] | None = None,
    ) -> AbstractContextManager[MutableMapping[str, object]]:
        """Yield a throwaway mapping and record nothing."""
        return nullcontext({})

    def shutdown(self) -> None:
        """Nothing to flush."""


class LangfuseTracer:
    """The real tracer: a Langfuse client plus its LangChain callback handler.

    Holds the client rather than reaching for ``langfuse.get_client()`` so the
    composition root stays the only place that decides tracing exists, and a
    test can hand in a client exporting to memory.
    """

    def __init__(
        self,
        client: Langfuse,
        *,
        public_key: str,
        trace_url_template: str | None = None,
    ) -> None:
        """Wrap ``client``; ``public_key`` is how a handler finds it again.

        ``trace_url_template`` is resolved once at startup by ``build_tracer``
        rather than per run, because the SDK's ``get_trace_url`` fetches the
        project id over HTTP: on the per-assessment path that is a blocking
        round trip, and a round trip that *fails slowly* whenever Langfuse is
        down -- the one moment tracing must cost nothing.
        """
        self._client = client
        self._public_key = public_key
        self._trace_url_template = trace_url_template

    def callbacks(self) -> list[BaseCallbackHandler]:
        """A handler for this run.

        Built per run rather than once and shared: the handler keeps a map of
        in-flight LangChain runs, and the API process can assess two claims
        concurrently. A fresh one costs an object allocation and removes the
        question entirely.
        """
        try:
            return [CallbackHandler(public_key=self._public_key)]
        except Exception:  # noqa: BLE001 - tracing never breaks a run
            logger.warning("tracing.handler_failed", exc_info=True)
            return []

    @contextmanager
    def assessment_run(
        self, *, assessment_id: str, correlation_id: str
    ) -> Iterator[None]:
        """Open the root span for one graph invocation and flush on the way out.

        The root span is ours rather than whichever runnable the callback
        handler happens to see first, which is what makes the trace id knowable
        *before* the run: it goes straight into a log line, so a correlation id
        in the logs leads to a trace in one hop. The correlation id also lands on
        the trace itself, as a tag and in the metadata, so the hop works the
        other way and in the search box.

        ``session_id`` is the assessment id, so the two halves of a paused
        assessment -- the run up to the human checkpoint and the run resumed
        after the decision -- group as one session in the UI.
        """
        stack = ExitStack()
        try:
            stack.enter_context(
                propagate_attributes(
                    session_id=assessment_id,
                    tags=[correlation_id],
                    metadata={
                        "correlation_id": correlation_id,
                        "assessment_id": assessment_id,
                    },
                    trace_name=TRACE_NAME,
                )
            )
            stack.enter_context(
                self._client.start_as_current_observation(
                    name=TRACE_NAME,
                    as_type="span",
                    input={
                        "assessment_id": assessment_id,
                        "correlation_id": correlation_id,
                    },
                )
            )
            trace_id = self._client.get_current_trace_id()
            logger.info(
                "trace.started",
                extra={
                    "trace_id": trace_id,
                    "trace_url": self._trace_url(trace_id),
                    "assessment_id": assessment_id,
                },
            )
        except Exception:  # noqa: BLE001 - tracing never breaks a run
            logger.warning("tracing.run_start_failed", exc_info=True)

        try:
            yield
        except Exception as exc:
            # Mark the root span before unwinding, so a failed run is red in the
            # UI instead of merely short.
            with suppress(Exception):
                self._client.update_current_span(
                    level="ERROR", status_message=f"{type(exc).__name__}: {exc}"
                )
            raise
        finally:
            with suppress(Exception):
                stack.close()
            # Per run, not per process: the worker is long-lived and a trace
            # nobody can read until the process exits is not an observability
            # tool. `flush` is a batched HTTP POST, not a per-span round trip.
            with suppress(Exception):
                self._client.flush()

    @contextmanager
    def span(
        self,
        name: str,
        *,
        input: Mapping[str, object],
        metadata: Mapping[str, object] | None = None,
    ) -> Iterator[MutableMapping[str, object]]:
        """Open a child of the current span; the yielded mapping becomes its output."""
        output: MutableMapping[str, object] = {}
        stack = ExitStack()
        observation = None
        try:
            observation = stack.enter_context(
                self._client.start_as_current_observation(
                    name=name,
                    as_type="span",
                    input=dict(input),
                    metadata=dict(metadata) if metadata is not None else None,
                )
            )
        except Exception:  # noqa: BLE001 - tracing never breaks a node
            logger.warning("tracing.span_failed", extra={"span": name}, exc_info=True)

        try:
            yield output
        finally:
            if observation is not None:
                with suppress(Exception):
                    observation.update(output=dict(output))
            with suppress(Exception):
                stack.close()

    def _trace_url(self, trace_id: str | None) -> str | None:
        """This run's trace URL, formatted locally from the startup template."""
        if trace_id is None or self._trace_url_template is None:
            return None
        return self._trace_url_template.replace(_TRACE_ID_PLACEHOLDER, trace_id)

    def shutdown(self) -> None:
        """Flush anything pending and stop the exporter."""
        with suppress(Exception):
            self._client.shutdown()


NO_TRACING: RunTracer = NullTracer()


def build_tracer(
    *,
    observability: ObservabilitySettings,
    llm: LlmSettings,
    span_exporter: SpanExporter | None = None,
) -> RunTracer:
    """Build the tracer this process should use, or the no-op one.

    ``span_exporter`` is the seam the unit tests use: an in-memory exporter
    gives a real Langfuse client, real spans and no server. Production passes
    nothing and the SDK exports over OTLP to ``langfuse_host``.
    """
    if not observability.tracing_active:
        logger.info(
            "tracing.disabled",
            extra={
                "tracing_enabled": observability.tracing_enabled,
                "has_credentials": bool(observability.langfuse_public_key),
            },
        )
        return NO_TRACING

    try:
        client = Langfuse(
            public_key=observability.langfuse_public_key,
            secret_key=observability.langfuse_secret_key.get_secret_value(),
            base_url=observability.langfuse_host,
            environment=observability.app_env,
            span_exporter=span_exporter,
        )
    except Exception:  # noqa: BLE001 - an unreachable tracer is not an outage
        logger.warning("tracing.client_failed", exc_info=True)
        return NO_TRACING

    register_model_prices(client, llm)
    logger.info("tracing.enabled", extra={"langfuse_host": observability.langfuse_host})
    return LangfuseTracer(
        client,
        public_key=observability.langfuse_public_key,
        trace_url_template=_resolve_trace_url_template(client),
    )


def _resolve_trace_url_template(client: Langfuse) -> str | None:
    """Learn the trace-URL shape once, so every run can format one for free.

    Asked of the SDK with a placeholder id rather than assembled here, so the
    URL layout stays Langfuse's business and a self-hosted instance behind a
    path prefix still produces links that work.
    """
    try:
        url = client.get_trace_url(trace_id=_TRACE_ID_PLACEHOLDER)
    except Exception:  # noqa: BLE001 - a missing link is not a failure
        logger.warning("tracing.trace_url_failed", exc_info=True)
        return None
    return url if url and _TRACE_ID_PLACEHOLDER in url else None


def model_price_definitions(llm: LlmSettings) -> list[tuple[str, float, float]]:
    """``(model name, USD per input token, USD per output token)`` for both models.

    Split out from the upsert so the arithmetic -- the only part that can be
    wrong without a server to tell you -- is directly testable.
    """
    return [
        (
            llm.llm_model_fast,
            llm.llm_fast_input_cost_per_1m_tokens_usd / _TOKENS_PER_PRICE_UNIT,
            llm.llm_fast_output_cost_per_1m_tokens_usd / _TOKENS_PER_PRICE_UNIT,
        ),
        (
            llm.llm_model_reasoning,
            llm.llm_reasoning_input_cost_per_1m_tokens_usd / _TOKENS_PER_PRICE_UNIT,
            llm.llm_reasoning_output_cost_per_1m_tokens_usd / _TOKENS_PER_PRICE_UNIT,
        ),
    ]


def register_model_prices(client: Langfuse, llm: LlmSettings) -> None:
    """Teach Langfuse what this project's models cost, so a trace shows a number.

    Langfuse prices a generation by matching the model name the callback handler
    reported against the model definitions in the project. It ships definitions
    for the well-known hosted models and knows nothing about an OpenRouter id
    like ``deepseek/deepseek-v4-flash-0731``, so without this every generation
    costs 0.00 and the DoD's "cost" is a blank column.

    Runs on every process start, so it has to be idempotent. Upserting on a
    deterministic id is not enough on its own: the model *name* is unique per
    project, so once a definition exists under some other id -- Langfuse assigns
    its own when it creates one -- a second upsert under ours is rejected with
    "already exists in project". So look the name up first and update whatever
    id it actually has, which also means a changed price in ``.env`` corrects the
    definition instead of being quietly ignored.

    Best effort throughout: a project that cannot be reached still traces, it
    just does not price.
    """
    definitions = [
        definition for definition in model_price_definitions(llm) if definition[0]
    ]
    if not definitions:
        return
    existing = _existing_model_ids(client, {name for name, _, _ in definitions})
    for model_name, input_price, output_price in definitions:
        try:
            client.api.models.upsert(
                existing.get(model_name, _model_definition_id(model_name)),
                model_name=model_name,
                # Anchored and escaped: the model id contains `/` and `-`, and a
                # loose pattern would silently price a different model.
                match_pattern=f"(?i)^{re.escape(model_name)}$",
                unit="TOKENS",
                # A pricing tier rather than the flat `input_price`/`output_price`
                # pair, which the API accepts only one of. The flat pair prices
                # the `input` and `output` usage keys and nothing else, and a
                # reasoning model reports its thinking under a third key,
                # `output_reasoning` -- 3725 of one measured compatibility call's
                # 4000 completion tokens. Providers bill those at the completion
                # rate, so pricing only input/output understates a reasoning
                # call's cost several-fold. See docs/OBSERVABILITY.md.
                pricing_tiers=[
                    PricingTierInput(
                        name="Standard",
                        is_default=True,
                        priority=0,
                        conditions=[],
                        prices={
                            "input": input_price,
                            "output": output_price,
                            "output_reasoning": output_price,
                        },
                    )
                ],
            )
        except Exception:  # noqa: BLE001 - pricing is a nicety, not a dependency
            logger.warning(
                "tracing.model_price_failed",
                extra={"model": model_name},
                exc_info=True,
            )


def _existing_model_ids(client: Langfuse, wanted: set[str]) -> dict[str, str]:
    """``model name -> definition id`` for the wanted names already in the project.

    Paged, because a fresh project already holds ~180 built-in definitions and
    the endpoint returns 50 at a time -- a first-page-only lookup would miss an
    existing definition and turn every restart into a failed upsert.
    """
    found: dict[str, str] = {}
    page = 1
    try:
        while wanted - found.keys():
            response = client.api.models.list(page=page, limit=_MODEL_PAGE_SIZE)
            for model in response.data:
                if model.model_name in wanted:
                    found[model.model_name] = model.id
            if page >= response.meta.total_pages:
                break
            page += 1
    except Exception:  # noqa: BLE001 - fall back to creating under our own id
        logger.warning("tracing.model_list_failed", exc_info=True)
    return found


def _model_definition_id(model_name: str) -> str:
    """A stable Langfuse model-definition id for ``model_name``."""
    return "claims-" + re.sub(r"[^a-z0-9]+", "-", model_name.lower()).strip("-")
