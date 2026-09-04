"""Run-scoped dependencies injected into every graph node -- [M4-01b].

``GraphContext`` is built once outside this package (the composition root, a
later issue) and handed to the graph as
``graph.invoke(state, context=GraphContext(...))``. Each node receives it as
``runtime.context`` by declaring a ``runtime: Runtime[GraphContext]`` parameter.
This is the only channel for the chat models, the retrieval port and the model
config -- a node holds no module-level client and takes no constructor. See
``docs/ARCHITECTURE.md`` ([M4-01b]).
"""

from collections.abc import Sequence
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
