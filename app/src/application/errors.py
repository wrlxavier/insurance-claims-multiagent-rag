"""The application layer's exception hierarchy [M5-02].

One module for every error a use case raises that is not a domain-invariant
violation (those live in ``domain.errors`` and stay there). These are the
failures the use cases detect *around* the domain rules -- an id that does not
resolve, a decision on an already-settled assessment, an edit that cites a
clause the corpus does not have, an orchestrator that broke its own contract.
[M5-04] maps each to an HTTP status with a stable error code; keeping them in
one place, mirroring ``domain.errors``, is what makes that mapping a single
``except`` group rather than a scattered set.

Standard library only -- the application layer imports no ``infrastructure``
(enforced by tests/architecture/test_layer_boundaries.py).
"""

from collections.abc import Iterable


class ApplicationError(Exception):
    """Base for every error the application layer raises."""


class AssessmentNotFoundError(ApplicationError):
    """No assessment exists for the given id.

    Carries the offending ``assessment_id`` so [M5-04] can report which id
    failed, mirroring ``domain.errors.CitationRequiredError``.
    """

    def __init__(self, assessment_id: str) -> None:
        """Build the message from the id that did not resolve."""
        self.assessment_id = assessment_id
        super().__init__(f"no assessment found for id {assessment_id!r}")


class AssessmentAlreadyDecidedError(ApplicationError):
    """A human decision was submitted for an assessment already settled."""

    def __init__(self, assessment_id: str) -> None:
        """Build the message from the id that already carries a decision."""
        self.assessment_id = assessment_id
        super().__init__(f"assessment {assessment_id!r} has already been decided")


class UnknownClauseError(ApplicationError):
    """An edited assessment cited one or more clauses absent from the corpus.

    Carries the missing ``clause_ids`` -- the reviewer's edit references
    something ``ClauseRepository`` cannot resolve.
    """

    def __init__(self, clause_ids: Iterable[str]) -> None:
        """Build the message from the clause ids that did not resolve."""
        self.clause_ids = tuple(clause_ids)
        joined = ", ".join(repr(cid) for cid in self.clause_ids)
        super().__init__(f"edited assessment cites unknown clause(s): {joined}")


class OrchestratorContractError(ApplicationError):
    """The orchestrator returned a result its port forbids.

    ``start`` must pause at the human checkpoint (``awaiting_review is True``);
    ``resume`` must run to completion (``awaiting_review is False``). A result
    that breaks either is an adapter bug, surfaced here rather than silently
    persisted.
    """
