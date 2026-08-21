"""The golden-set question schema [M2-01].

Fixes the format every golden question must follow, so question 90 is
comparable to question 1. Mirrors [infrastructure.parsing.clause_schema]'s
``ParsedClauseRecord`` pattern: a flat, validated Pydantic row that fails
loudly on a malformed question rather than accepting it silently.

Authorship is a three-layer flow, not a single step: the author selects the
source clause first, from the parsed M1 clause tree (never the raw PDF);
the LLM drafts the question's phrasing from that clause; the author then
verifies both correctness (does the referenced clause actually answer the
question?) and completeness (is any clause missing from
``reference_clause_ids``?) before the question is accepted. See
``docs/EVALUATION.md`` for the full authoring rules.
"""

from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field, model_validator

SCHEMA_VERSION = "v1"

_NonEmptyStr = Annotated[str, Field(min_length=1)]


class QuestionType(Enum):
    """What kind of retrieval/reasoning task a golden question exercises."""

    DIRECT_LOOKUP = "direct_lookup"
    COVERAGE_WITH_EXCLUSION = "coverage_with_exclusion"
    CROSS_DOCUMENT = "cross_document"
    UNANSWERABLE = "unanswerable"
    DEFINITION = "definition"


class Difficulty(Enum):
    """How hard the retrieval/reasoning step is within a question_type."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class ExpectedVerdict(Enum):
    """The project's one verdict vocabulary, reused verbatim.

    Canonical source: ``SCOPE_PREAMBLE`` in
    [infrastructure.graph.prompts.scope_preamble] -- "Every verdict you
    produce must be exactly one of: compatible, incompatible,
    insufficient_information."
    """

    COMPATIBLE = "compatible"
    INCOMPATIBLE = "incompatible"
    INSUFFICIENT_INFORMATION = "insufficient_information"


class GoldenQuestion(BaseModel):
    """One golden-set question row.

    ``unanswerable`` questions must carry an empty ``reference_clause_ids``
    (nothing answers them, by construction) and
    ``expected_verdict=insufficient_information``; every other
    ``question_type`` must carry at least one reference clause id -- a
    question needs something to point at, and reference clauses are
    exhaustive, so a retriever that returns only half of them is measurably
    wrong.
    """

    schema_version: str
    question_id: _NonEmptyStr
    document_id: _NonEmptyStr
    question: _NonEmptyStr
    reference_clause_ids: list[str]
    question_type: QuestionType
    difficulty: Difficulty
    expected_verdict: ExpectedVerdict | None
    notes: str = ""
    authored_at: str | None = None

    @model_validator(mode="after")
    def _check_unanswerable_consistency(self) -> "GoldenQuestion":
        if self.question_type == QuestionType.UNANSWERABLE:
            if self.reference_clause_ids:
                raise ValueError(
                    "unanswerable questions must have empty reference_clause_ids, "
                    f"got {self.reference_clause_ids!r}"
                )
            if self.expected_verdict != ExpectedVerdict.INSUFFICIENT_INFORMATION:
                raise ValueError(
                    "unanswerable questions must have "
                    "expected_verdict=insufficient_information, "
                    f"got {self.expected_verdict!r}"
                )
        elif not self.reference_clause_ids:
            raise ValueError(
                f"{self.question_type.value} questions must have at least one "
                "reference_clause_ids entry"
            )
        return self
