"""The demo helper API -- three read-only endpoints under ``/demo/api`` [M6-03].

Not part of the assessment API contract (``docs/API.md`` / ``docs/DEMO_UI.md``).
The SPA drives the real ``/v1`` endpoints for everything else; these three cover
what ``/v1`` deliberately does not: the curated examples, the citation ->
full-clause join, and a live pipeline-progress read.

No ``Depends`` and no ``app.state`` coupling -- the router reads static files and
(for progress) opens its own checkpointer connection, so it is exercised by a
bare ``TestClient(create_app())``.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query

from presentation.demo.clause_context import clause_index
from presentation.demo.examples import load_examples
from presentation.demo.progress import PIPELINE, read_progress
from presentation.demo.schemas import (
    ClauseContextResponse,
    DemoExample,
    ProgressResponse,
)

router = APIRouter(prefix="/demo/api", tags=["demo"])


@router.get("/examples", response_model=list[DemoExample])
def list_examples() -> list[DemoExample]:
    """The curated one-click example claims."""
    return load_examples()


@router.get("/clause-context", response_model=ClauseContextResponse)
def clause_context(
    clause_id: Annotated[list[str], Query(min_length=1)],
) -> ClauseContextResponse:
    """Join citation ``clause_id``s to full clause text, page span and source label.

    ``index_available`` is False when ``build/parsed_clauses.jsonl`` could not be
    read; the SPA then renders citations from the ``/v1`` excerpt alone.
    """
    index = clause_index()
    return ClauseContextResponse(
        index_available=bool(index),
        contexts={cid: index[cid] for cid in clause_id if cid in index},
        missing=[cid for cid in clause_id if cid not in index],
    )


@router.get("/assessments/{assessment_id}/progress", response_model=ProgressResponse)
def assessment_progress(assessment_id: str) -> ProgressResponse:
    """Where ``assessment_id`` sits in the pipeline, read live from the checkpoint."""
    view = read_progress(assessment_id)
    return ProgressResponse(
        available=view.available,
        started=view.started,
        pipeline=list(PIPELINE),
        nodes_seen=view.nodes_seen,
        next_nodes=view.next_nodes,
        interrupted=view.interrupted,
    )
