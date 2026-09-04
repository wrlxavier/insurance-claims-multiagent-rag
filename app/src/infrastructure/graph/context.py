"""Run-scoped dependencies injected into every graph node -- [M4-01b].

``GraphContext`` is built once outside this package (the composition root, a
later issue) and handed to the graph as
``graph.invoke(state, context=GraphContext(...))``. Each node receives it as
``runtime.context`` by declaring a ``runtime: Runtime[GraphContext]`` parameter.
This is the only channel for the chat models, the retrieval port and the model
config -- a node holds no module-level client and takes no constructor. See
``docs/ARCHITECTURE.md`` ([M4-01b]).
"""

from collections.abc import Mapping, MutableMapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from langchain_core.language_models.chat_models import BaseChatModel

from infrastructure.config.settings import LlmSettings
from infrastructure.graph.state import AuditRecord

if TYPE_CHECKING:
    from infrastructure.rag.retrieval_filter import RetrievalFilter
    from infrastructure.rag.retrieved_clause import RetrievedClause


class RetrievalPort(Protocol):
    """The retrieval capability a node depends on, owned by the graph layer.

    The signature mirrors the M3 retrievers
    (``infrastructure.evaluation.retriever.FilterableRetriever``) so a concrete
    hybrid/rerank retriever satisfies it structurally, without the graph
    importing a concrete implementation or the evaluation harness's interface
    ([M4-04]). ``retrieve`` returns ranked, scored, hydrated
    ``infrastructure.rag.retrieved_clause.RetrievedClause`` rows -- the retrieval
    node ([M4-04]) maps each to a ``state.Citation`` and assembles the [M3-07]
    ``GateSignals`` from the scores. A bare-clause-id retriever does not satisfy
    this port; [infrastructure.rag.graph_retrieval_adapter.GraphRetrievalAdapter]
    is the adapter that does. [M4-04] owns any further change to this shape.
    """

    def retrieve(
        self,
        question: str,
        *,
        k: int,
        metadata_filter: "RetrievalFilter | None" = None,
    ) -> "list[RetrievedClause]":
        """Return up to ``k`` clauses, ranked best-match first, with provenance."""
        ...


class AuditTrailSink(Protocol):
    """Durable storage for the audit trail, owned by the graph layer -- [M4-09].

    Graph state is already persisted by the checkpointer, but only LangGraph can
    read it back. The audit trail is the record a compliance reader has to be
    able to query directly, so it is written once more, to a table of its own.

    ``thread_id`` plus the record's position in ``records`` is the identity of a
    row: the write is expected to be idempotent, because the checkpoint node
    that calls this runs again whenever its thread is resumed.
    [infrastructure.database.graph_audit_sink.SqlAlchemyAuditTrailSink] is the
    implementation; the port lives here for the same reason ``RetrievalPort``
    does -- a node depends on the capability, never on the adapter, and the
    adapter satisfies this structurally without importing it.
    """

    def record(
        self,
        *,
        claim_id: str,
        thread_id: str,
        records: Sequence[AuditRecord],
    ) -> int:
        """Persist ``records`` for one graph run. Return the number of new rows."""
        ...


class TracePort(Protocol):
    """Somewhere to open one span of a run, owned by the graph layer -- [M5-07].

    Most of the trace costs a node nothing: the Langfuse callback handler the
    orchestrator installs on the graph config already opens a span per node and
    a generation per LLM call. This port exists for the one thing callbacks
    cannot see -- work a node does *between* those calls, where the interesting
    numbers are locals. The retrieval node ([M4-04]) is the case that motivates
    it: its candidates, their scores and the [M3-07] gate's reasoning never
    appear in an LLM call and mostly never reach state either.

    Deliberately dumb. It deals in plain mappings, not domain objects, so the
    graph layer grows no tracing vocabulary and a test fake is a handful of
    lines. It is a context manager rather than a "record this finished span"
    call because the span's latency should be measured, not reported: the
    yielded mapping is the span's output, filled in by the body.

        with runtime.context.tracer.span("retrieval", input={...}) as traced:
            hits = ...
            traced["n_returned"] = len(hits)

    [infrastructure.observability.tracing.LangfuseTracer] is the
    implementation; it satisfies this structurally, exactly as ``RetrievalPort``
    and ``AuditTrailSink`` are satisfied.

    An implementation **must not raise**: a run is not allowed to fail because
    its observability did.
    """

    def span(
        self,
        name: str,
        *,
        input: Mapping[str, object],
        metadata: Mapping[str, object] | None = None,
    ) -> AbstractContextManager[MutableMapping[str, object]]:
        """Open a span named ``name``; fill the yielded mapping to set its output."""
        ...


class _NoTracing:
    """The do-nothing ``TracePort``: what a node gets when tracing is off."""

    def span(
        self,
        name: str,
        *,
        input: Mapping[str, object],
        metadata: Mapping[str, object] | None = None,
    ) -> AbstractContextManager[MutableMapping[str, object]]:
        """Yield a throwaway mapping and record nothing."""
        return nullcontext({})


# A null object rather than the ``| None`` shape ``audit_sink`` uses below. The
# audit sink is consulted once, at the checkpoint; a tracer wraps work inside a
# node body, and a null object keeps that body free of `if tracer is not None`
# noise for a call whose whole point is that it changes nothing.
NO_TRACING: TracePort = _NoTracing()


@dataclass(frozen=True)
class GraphContext:
    """Run-scoped dependencies for the agent graph -- [M4-01b].

    Frozen: a node reads these, it does not swap them. The two chat models are
    pre-built so a unit test passes fakes directly. ``llm_settings`` is the
    model config -- a node records ``llm_settings.llm_model_reasoning`` (or
    ``.llm_model_fast``) as the ``AuditEvent.model`` for a call it makes,
    rather than introspecting the model object. The OpenRouter provider pins
    are applied by ``build_chat_model`` at the composition root, never read by
    a node.
    """

    fast_model: BaseChatModel
    reasoning_model: BaseChatModel
    retriever: RetrievalPort
    llm_settings: LlmSettings
    # [M4-09]. Defaulted, so a context built for a test or an eval script needs
    # no durable store; `None` means "this run leaves no separate audit record",
    # the same way a deterministic node leaves `AuditEvent.model` unset. The
    # composition root supplies the Postgres-backed sink.
    audit_sink: AuditTrailSink | None = None
    # [M5-06]. The correlation id for this run -- tied to the originating HTTP
    # request (or minted by the worker). `infrastructure.graph.build`'s node
    # wrapper stamps it on every node log line; the orchestrator also puts it in
    # the graph `config` metadata so LLM calls inherit it. Defaulted so a test
    # or eval `GraphContext(...)` needs nothing.
    correlation_id: str = ""
    # [M5-07]. Where a node records a span the callback-handler trace cannot see
    # -- see ``TracePort``. Defaults to the no-op, so every test, eval script and
    # unconfigured deployment builds a ``GraphContext`` exactly as before and the
    # nodes take the same path either way.
    tracer: TracePort = NO_TRACING
