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
  ``_build_filter``: the SUSEP process when the claim stated one, else the
  product line, else an unconstrained search (the [M3-04] unknown-process
  degradation path). The two are deliberately *not* ANDed; [M4-10] measured what
  that cost, and ``_build_filter``'s docstring has the reason. The graph has no
  insurer CNPJ from intake, so this filter is one field weaker than the M3 eval
  harness's SUSEP-process + CNPJ default path -- see ``docs/RETRIEVAL_NODE.md``;
- calls the port, maps each ``RetrievedClause`` to a typed ``state.Citation``;
- assembles the [infrastructure.rag.insufficient_context_gate.GateSignals] and
  runs [evaluate_gate] to set ``context_sufficient``. The router
  ``route_after_retrieval`` in ``build.py`` acts on that flag.

[M5-07] wraps that in one explicit span. Retrieval is the only node the trace's
callback handler sees as a black box, because it makes no LLM call: what it
retrieved, at what scores, under which filter, and why the gate decided as it
did are all locals here, and most of them never reach state. They are the first
thing you want when a verdict is wrong, so the span records them -- including
the three ``InsufficientContextResult`` fields (``threshold``,
``missing_category``, ``closest_clause_ids``) that are computed on every run and
otherwise thrown away.
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

    with context.tracer.span(
        "retrieval",
        input={
            "query": query,
            "k": RETRIEVAL_K,
            "filter": _filter_payload(metadata_filter),
        },
    ) as traced:
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

        traced.update(
            {
                "n_returned": len(hits),
                # Ranked, scored, and named -- the span's reason to exist. Read
                # top to bottom this is what the assessment nodes will see.
                "candidates": [
                    {
                        "rank": rank,
                        "clause_id": hit.clause_id,
                        "document_id": hit.document_id,
                        "susep_process": hit.susep_process,
                        "clause_type": hit.clause_type.value,
                        "score": hit.score,
                    }
                    for rank, hit in enumerate(hits, start=1)
                ],
                "gate": {
                    "sufficient": gate.sufficient,
                    "trigger": gate.trigger.value or None,
                    "top_score": gate.top_score,
                    "threshold": gate.threshold,
                    "missing_category": gate.missing_category.value
                    if gate.missing_category is not None
                    else None,
                    "closest_clause_ids": list(gate.closest_clause_ids),
                    "explanation": gate.explanation,
                },
            }
        )

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


def _filter_payload(metadata_filter: RetrievalFilter | None) -> dict[str, str | None]:
    """The metadata pre-filter as trace-friendly fields, or ``None`` for unfiltered."""
    if metadata_filter is None:
        return {"susep_process": None, "product_line": None}
    return {
        "susep_process": metadata_filter.susep_process,
        "product_line": metadata_filter.product_line,
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

    A stated SUSEP process wins **alone**; the classified product line
    constrains only the fallback, when the claim named no process. Returns
    ``None`` when neither is known -- the [M3-04] all-``None`` degradation path
    is an unconstrained search, not an error.

    **Why the two are not ANDed** ([M4-10]). They do not describe the same
    thing: a process names the registered product the claim was *filed
    against*, while ``product_line`` is intake's classification of the *event
    the claimant described*. On a product/claim mismatch those disagree by
    construction, so the conjunction selects nothing at all, the [M3-07] gate
    fires on an empty result, and a knowable ``incompatible`` degrades to
    ``insufficient_information``. Measured: under the conjunction all 11 of
    ``product_claim_mismatch.jsonl``'s claims had an empty search space, and so
    would 9 of the 13 documents the main claim set targets whenever intake reads
    the event as CASCO. The process is also the stricter of the two -- it names
    one document, where a product line names a whole segment of the corpus.
    See ``docs/END_TO_END_EVALUATION.md`` and ``docs/ARCHITECTURE.md``.
    """
    if entities is None:
        return None
    if entities.susep_process:
        return RetrievalFilter(susep_process=entities.susep_process)
    candidate = RetrievalFilter(product_line=entities.product_line)
    return None if candidate.is_empty else candidate
