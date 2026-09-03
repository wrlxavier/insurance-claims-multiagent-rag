"""The concrete ``ClaimAssessmentOrchestrator`` -- LangGraph behind the port [M5-04].

``application.ports.claim_assessment_orchestrator.ClaimAssessmentOrchestrator``
promises the application layer an assessment without a graph in sight. This is
the adapter that keeps that promise: it compiles ``build_claim_graph()`` against
the Postgres checkpointer, runs a claim to the human checkpoint (``start``) or
resumes a paused run to completion (``resume``), and hands back a graph-free
``OrchestratorResult`` via ``infrastructure.graph.state_mapper``.

Design notes:

- **``assessment_id`` is the graph ``thread_id``** -- the port's sole run key, so
  ``resume`` needs nothing more to find the paused run.
- **A checkpointer connection per call.** ``open_claim_checkpointer`` opens its
  own autocommit psycopg connection; a fresh SQLAlchemy session per run feeds the
  retriever. Concurrent requests never share either. A pooled connection is
  "the M5 shape" (``docs/DATABASE.md``); [M5-05]'s worker model revisits it.
- **The audit trail is captured, not committed here.** The composition root
  builds the per-run ``GraphContext`` without a sink; this adapter swaps in a
  ``_CapturingAuditSink`` so the ``human_review`` node's trail comes back in the
  ``OrchestratorResult`` and ``SubmitHumanDecision`` writes it in the same
  transaction as the settled record (``docs/ARCHITECTURE.md``, the [M5-04] fold).
- **The contract checks belong to the use case.** ``start`` not pausing / ``resume``
  not finishing is surfaced by the use case as ``OrchestratorContractError``
  (per the port docstring); this adapter just reports ``awaiting_review`` as it
  found it and still returns a well-formed result either way.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any, cast

from langgraph.graph import StateGraph
from langgraph.types import Command
from sqlalchemy.orm import Session, sessionmaker

from application.orchestrator_result import OrchestratorResult
from domain.claim import Claim
from domain.human_decision import HumanDecision
from infrastructure.graph import state_mapper
from infrastructure.graph.build import build_claim_graph
from infrastructure.graph.checkpointer import open_claim_checkpointer
from infrastructure.graph.context import GraphContext
from infrastructure.graph.state import AuditRecord, ClaimState

_GraphFactory = Callable[
    [], StateGraph[ClaimState, GraphContext, ClaimState, ClaimState]
]
_ContextFactory = Callable[[Session], GraphContext]


class _CapturingAuditSink:
    """An ``AuditTrailSink`` that keeps the run's trail in memory for the adapter.

    ``human_review`` calls ``record`` once, after the pause, with the run's whole
    trail. The adapter reads it back and returns it as
    ``OrchestratorResult.audit_records`` for the use case to persist
    transactionally -- so nothing is committed from inside the graph node.
    """

    def __init__(self) -> None:
        self.records: list[AuditRecord] = []
        self.calls: list[tuple[str, str]] = []

    def record(
        self,
        *,
        claim_id: str,
        thread_id: str,
        records: Sequence[AuditRecord],
    ) -> int:
        """Capture ``records`` (the whole trail) and report how many there are."""
        self.records = list(records)
        self.calls.append((claim_id, thread_id))
        return len(self.records)


class LangGraphClaimAssessmentOrchestrator:
    """Run and resume claim assessments on the compiled LangGraph graph."""

    def __init__(
        self,
        *,
        context_factory: _ContextFactory,
        session_factory: sessionmaker[Session],
        database_url: str | None = None,
        graph_builder: _GraphFactory = build_claim_graph,
    ) -> None:
        """Wire the adapter to its per-run dependencies.

        ``context_factory`` builds a ``GraphContext`` on a fresh session (models
        and retriever components are singletons the composition root closes over);
        ``database_url`` is passed to ``open_claim_checkpointer`` -- ``None`` uses
        the service database the rest of the app reads.
        """
        self._context_factory = context_factory
        self._session_factory = session_factory
        self._database_url = database_url
        self._graph_builder = graph_builder

    def start(self, *, assessment_id: str, claim: Claim) -> OrchestratorResult:
        """Assess ``claim`` from scratch, up to the human checkpoint."""
        final_state, sink = self._invoke(
            assessment_id,
            {
                "claim_id": claim.claim_id,
                "raw_claim_text": self._claim_text(claim),
            },
        )
        return state_mapper.result_from_final_state(
            final_state,
            awaiting_review="__interrupt__" in final_state,
            audit_records=state_mapper.audit_entries_from_records(sink.records),
        )

    def resume(
        self, *, assessment_id: str, decision: HumanDecision
    ) -> OrchestratorResult:
        """Resume the paused run ``assessment_id`` with the analyst's decision."""
        final_state, sink = self._invoke(
            assessment_id,
            Command(resume=state_mapper.resume_payload(decision)),
        )
        return state_mapper.result_from_final_state(
            final_state,
            awaiting_review="__interrupt__" in final_state,
            audit_records=state_mapper.audit_entries_from_records(sink.records),
        )

    def _invoke(
        self,
        assessment_id: str,
        graph_input: ClaimState | Command[Any],
    ) -> tuple[dict[str, Any], _CapturingAuditSink]:
        """Compile the graph, invoke it once for this run, return its final state."""
        config: Any = {"configurable": {"thread_id": assessment_id}}
        sink = _CapturingAuditSink()
        with (
            self._session_factory() as session,
            open_claim_checkpointer(self._database_url) as checkpointer,
        ):
            compiled = self._graph_builder().compile(checkpointer=checkpointer)
            context = replace(self._context_factory(session), audit_sink=sink)
            final_state = compiled.invoke(graph_input, config=config, context=context)
        return cast("dict[str, Any]", final_state), sink

    @staticmethod
    def _claim_text(claim: Claim) -> str:
        """The graph's ``raw_claim_text``: the narrative, with its policy if known.

        A one-line SUSEP-process header the way a claim filed against a known
        policy would carry it -- byte-identical to
        ``scripts/eval_end_to_end.py::build_claim_text``'s measured headline arm,
        so intake extracts the process and ``nodes/retrieval._build_filter``
        pre-filters on it with no graph change and no eval regression.
        """
        if claim.policy_ref is None:
            return claim.raw_text
        return (
            f"[Apólice registrada: processo SUSEP {claim.policy_ref.value}]\n"
            f"{claim.raw_text}"
        )
