"""Types for [M1-04d]'s vision-LLM boundary-escalation pass.

Kept separate from [domain.clause_tree], the same way [domain.
clause_classification] keeps [TypedClause] separate from [Clause] itself:
this is a second, independent pass over an already-built [ClauseTree], not
a property of tree construction.

Frozen dataclasses only, no third-party imports -- same stdlib-only
constraint as [domain.clause_tree] and [domain.clause_classification].
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SuspicionFlag:
    """One clause the deterministic pass flags as worth a vision review.

    ``reasons`` names every matched trigger (a clause can be both oversized
    and OCR-derived, for instance) -- see [application.use_cases.
    boundary_escalation.find_suspicious_clauses] for the exact predicate.
    """

    clause_id: str
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class BoundaryReview:
    """The vision model's judgment for one flagged clause's boundary.

    ``corrected_page_start``/``corrected_page_end`` are only meaningful
    when ``confirmed`` is False; ``split_suggested``/``split_notes`` are a
    separate, independent signal -- a model can confirm the *outer*
    boundary while still suggesting an internal split.
    """

    confirmed: bool
    corrected_page_start: int | None
    corrected_page_end: int | None
    split_suggested: bool
    split_notes: str
    reasoning: str


@dataclass(frozen=True)
class BoundaryEscalationOutcome:
    """What happened for one flagged clause after review.

    ``applied`` is False whenever the review was a no-op confirmation, a
    split suggestion (never auto-applied -- see the module docstring in
    [application.use_cases.boundary_escalation]), or a correction that
    could not be safely applied (falls outside the rasterized margin, or
    the clause/its neighbor lacks per-line page attribution). ``note``
    always explains which case it was.
    """

    clause_id: str
    reasons: tuple[str, ...]
    review: BoundaryReview
    applied: bool
    note: str
