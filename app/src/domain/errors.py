"""The domain layer's exception hierarchy [M5-01].

One module for every error the business rules raise, rather than the
per-module exceptions used elsewhere in ``domain/`` (``OrphanTextExceeds
ThresholdError`` and friends): those are loud-failure guards for a single
script path, whereas M5-01 introduces a *cluster* of construction
invariants that the application layer ([M5-02]) needs to catch as one group
at the API boundary and map to HTTP status codes. A scattered set makes
that fragile.

Standard library only, like the rest of ``domain/`` -- enforced by
tests/architecture/test_layer_boundaries.py.
"""


class DomainError(Exception):
    """Base for every error the domain layer raises."""


class InvalidValueObjectError(DomainError, ValueError):
    """A value object was constructed from a malformed value.

    Also a :class:`ValueError` so a caller can ``except ValueError`` the way
    Python code conventionally does for a bad constructor argument, while
    still being able to catch the precise domain type.
    """


class InvalidSusepProcessError(InvalidValueObjectError):
    """A SUSEP process number did not match ``NNNNN.NNNNNN/NNNN-NN``."""


class InvalidCnpjError(InvalidValueObjectError):
    """A CNPJ was not 14 digits, or its check digits did not verify."""


class InvariantViolationError(DomainError):
    """An entity was constructed in a state its business rules forbid."""


class CitationRequiredError(InvariantViolationError):
    """An assessment was built with no citations.

    The one M5-01 invariant with a dedicated type: it carries the offending
    ``assessment_id`` so [M5-02] can report *which* assessment failed,
    mirroring ``domain.clause_classification.MissingProvenanceError``.
    """

    def __init__(self, assessment_id: str) -> None:
        """Build the message from the citation-free assessment's id."""
        self.assessment_id = assessment_id
        super().__init__(
            f"assessment {assessment_id!r} must carry at least one citation"
        )


class VerdictNotPermittedError(InvariantViolationError):
    """An assessment's verdict was not a :class:`domain.verdict.Verdict` member."""


class DecisionMustReferenceAssessmentError(InvariantViolationError):
    """A human decision was recorded without the id of the assessment it acted on."""


class PolicyYearMismatchError(InvariantViolationError):
    """A policy's ``process_year`` disagreed with its SUSEP process number."""
