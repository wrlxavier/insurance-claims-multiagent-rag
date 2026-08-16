"""LLM Implementation of the ClauseClassifierPort."""

# Assuming langchain integration as requested by the stack context
from langchain_core.language_models.chat_models import BaseChatModel  # type: ignore
from langchain_core.prompts import ChatPromptTemplate  # type: ignore
from pydantic import BaseModel, Field

from application.ports.clause_classifier import ClauseClassifierPort
from domain.clause_classification import ClauseType


class LLMClassificationOutput(BaseModel):
    """Structured output expected from the LLM."""

    clause_type: ClauseType = Field(
        ..., description="The classified type of the clause."
    )
    confidence: float = Field(..., description="Confidence score from 0.0 to 1.0.")


class LangchainClauseClassifier(ClauseClassifierPort):
    """Uses a Langchain BaseChatModel with structured output to classify clauses."""

    def __init__(self, llm: BaseChatModel) -> None:
        """Initialize the classifier with a Langchain model."""
        self.llm = llm.with_structured_output(LLMClassificationOutput)
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a Brazilian insurance clause classification expert. "
                    "Classify the provided insurance policy clause into exactly "
                    "one of the following types: COVERAGE, EXCLUSION, CONDITION, "
                    "DEFINITION, PROCEDURE, OTHER. Provide a confidence score "
                    "between 0.0 and 1.0.",
                ),
                ("human", "Clause Title: {title}\\n\\nClause Text: {text}"),
            ]
        )

    def classify(self, clause_title: str, clause_text: str) -> tuple[ClauseType, float]:
        """Classify the given clause.

        Args:
            clause_title: The original un-normalized title.
            clause_text: The full text of the clause.

        Returns:
            A tuple of (ClauseType, confidence).
        """
        chain = self.prompt | self.llm
        result: LLMClassificationOutput = chain.invoke(
            {"title": clause_title, "text": clause_text}
        )
        return result.clause_type, result.confidence
