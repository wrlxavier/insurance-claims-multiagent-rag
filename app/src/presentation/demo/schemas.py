"""Response models for the demo helper endpoints -- [M6-03].

Pydantic v2, like ``presentation.schemas``: the demo router speaks JSON, so this
is where its shapes live. None of these cross into ``domain`` / ``application``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

_CATEGORIES = Literal[
    "compatible",
    "incompatible",
    "insufficient_information",
    "product_claim_mismatch",
]
_VERDICTS = Literal["compatible", "incompatible", "insufficient_information"]


class DemoExample(BaseModel):
    """One curated one-click claim, authored from the synthetic set.

    ``raw_text`` is the narrative copied verbatim from
    ``data/synthetic_claims/*.jsonl``; ``source_claim_id`` names the row it came
    from so a test can assert the copy never drifted. ``expected_verdict`` is
    what the run should reach -- not a guarantee (a live run is
    non-deterministic).
    """

    id: str
    label: str
    category: _CATEGORIES
    expected_verdict: _VERDICTS
    note: str
    raw_text: str
    policy_ref: str | None = None
    claim_id: str | None = None
    source_claim_id: str


class ClauseContext(BaseModel):
    """The full-clause view behind one citation, from ``build/parsed_clauses.jsonl``.

    The ``/v1`` citation object carries only a short ``excerpt`` and ids; this
    adds the whole clause text, its page span in the source filing, and a
    human-readable source label -- the DoD's "clause text with its source
    document and page".
    """

    clause_id: str
    document_id: str
    title: str
    text: str
    page_start: int
    page_end: int
    clause_type: str
    susep_process: str
    insurer: str
    product_line: str
    filing_year: str


class ClauseContextResponse(BaseModel):
    """The batch clause-context lookup for a review screen's citations."""

    index_available: bool
    contexts: dict[str, ClauseContext]
    missing: list[str]


class ProgressResponse(BaseModel):
    """A live read of one run's position in the pipeline, from the checkpoint.

    ``available`` is False when the checkpoint could not be read at all (the SPA
    then falls back to a plain elapsed timer); ``started`` is False when the read
    worked but no checkpoint exists for the id yet (the worker has not begun).
    ``nodes_seen`` is which graph nodes have recorded an audit event so far,
    ``next_nodes`` what LangGraph will run next, ``interrupted`` whether the run
    is paused at the human checkpoint.
    """

    available: bool
    started: bool
    pipeline: list[str]
    nodes_seen: list[str]
    next_nodes: list[str]
    interrupted: bool
