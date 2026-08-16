"""Port for the LLM clause classification."""

from typing import Protocol

from domain.clause_classification import ClauseType


class ClauseClassifierPort(Protocol):
    """Interface for classifying clauses when deterministic rules fail."""

    def classify(self, clause_title: str, clause_text: str) -> tuple[ClauseType, float]:
        """Classify the given clause.

        Args:
            clause_title: The original un-normalized title.
            clause_text: The full text of the clause.

        Returns:
            A tuple of (ClauseType, confidence).
        """
        ...
