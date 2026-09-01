"""The retrieval node: entities -> retrieved clauses + the sufficiency flag ([M4-04]).

Wraps the M3 pipeline behind [infrastructure.graph.context.RetrievalPort]. The
node itself is deterministic -- no LLM call -- so its ``AuditEvent`` leaves
``model`` / ``token_usage`` / ``confidence`` as ``None`` (like
``nodes/clarification_exhausted.py``), and it needs no ``schemas.py`` entry and
no prompt file.

What it does:

- builds the retrieval query from the extracted *entities*, not the raw claim
  text ([M4-04] DoD) -- ``_build_query``;
- builds the metadata pre-filter from intake's classification ([M4-04] DoD) --
  ``_build_filter``: SUSEP process when stated plus the product line, or an
  unconstrained search (the [M3-04] unknown-process degradation path). The graph
  has no insurer CNPJ from intake, so this filter is one field weaker than the
  M3 eval harness's SUSEP-process + CNPJ default path -- see
  ``docs/RETRIEVAL_NODE.md``;
- calls the port, maps each ``RetrievedClause`` to a typed ``state.Citation``;
- assembles the [infrastructure.rag.insufficient_context_gate.GateSignals] and
  runs [evaluate_gate] to set ``context_sufficient``. The router
  ``route_after_retrieval`` in ``build.py`` acts on that flag.
"""

from __future__ import annotations

from langgraph.runtime import Runtime

from infrastructure.graph.context import GraphContext
from infrastructure.graph.state import (
    AuditEvent,
    Citation,
    ClaimState,
    ExtractedEntities,
)
from infrastructure.rag.insufficient_context_gate import GateSignals, evaluate_gate
from infrastructure.rag.retrieval_filter import RetrievalFilter

# How many clauses the node asks the retriever for. Product behaviour defined in
# code (like ``build.MAX_CLARIFICATION_ROUNDS``), not a deployment knob: it
# matches the [M3-05] rerank candidate depth the [M3-07] gate was calibrated at,
# so the gate's top-score signal here is the one the calibration measured.
RETRIEVAL_K = 10

_NODE_INPUT_PREVIEW_CHARS = 180


def retrieval(state: ClaimState, runtime: Runtime[GraphContext]) -> dict[str, object]:
    """Retrieve clauses for the claim and flag whether the context suffices."""
    context = runtime.context
    entities = state.get("entities")

    query = _build_query(entities, fallback=state["raw_claim_text"])
    metadata_filter = _build_filter(entities)

    hits = context.retriever.retrieve(
        query, k=RETRIEVAL_K, metadata_filter=metadata_filter
    )

    citations = [
        Citation(
            clause_id=hit.clause_id,
            document_id=hit.document_id,
            susep_process=hit.susep_process,
            clause_type=hit.clause_type,
            relevance_score=max(hit.score, 0.0),
            excerpt=hit.excerpt,
        )
        for hit in hits
    ]

    signals = GateSignals(
        top_score=hits[0].score if hits else 0.0,
        reranked_scores=tuple(hit.score for hit in hits),
        retrieved_clause_ids=tuple(hit.clause_id for hit in hits),
        retrieved_clause_types=tuple(hit.clause_type for hit in hits),
        k_requested=RETRIEVAL_K,
        n_returned=len(hits),
    )
    gate = evaluate_gate(query, signals)

    audit_event = AuditEvent(
        node="retrieval",
        action="retrieve_clauses",
        node_input=(
            f"query={query[:_NODE_INPUT_PREVIEW_CHARS]!r} "
            f"n_returned={len(hits)} sufficient={gate.sufficient} "
            f"trigger={gate.trigger.value or 'none'}"
        ),
    )
    return {
        "citations": citations,
        "context_sufficient": gate.sufficient,
        "audit_trail": [audit_event],
    }


def _build_query(entities: ExtractedEntities | None, *, fallback: str) -> str:
    """Assemble the retrieval query from the extracted entities.

    Joins the non-empty of ``event_type`` / ``description`` / ``vehicle_info``.
    Falls back to ``fallback`` (the raw claim text) only when that leaves nothing
    -- a claim with no usable entities routes to clarification, not here, so the
    fallback is defensive rather than the normal path.
    """
    if entities is None:
        return fallback
    values = (entities.event_type, entities.description, entities.vehicle_info)
    parts = [value.strip() for value in values if value and value.strip()]
    return " ".join(parts) if parts else fallback


def _build_filter(entities: ExtractedEntities | None) -> RetrievalFilter | None:
    """Build the metadata pre-filter from intake's classification.

    SUSEP process (only when intake actually read one from the claim) plus the
    classified product line. Returns ``None`` when neither is known -- the
    [M3-04] all-``None`` degradation path is an unconstrained search, not an
    error.
    """
    if entities is None:
        return None
    candidate = RetrievalFilter(
        susep_process=entities.susep_process,
        product_line=entities.product_line,
    )
    return None if candidate.is_empty else candidate
