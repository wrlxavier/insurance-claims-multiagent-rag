from typing import Any, cast

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable, RunnableLambda

from domain.clause_classification import ClauseType
from infrastructure.parsing.llm_classifier import (
    LangchainClauseClassifier,
    LLMClassificationOutput,
)


class FakeChatModel:
    """Stand-in for a Langchain ``BaseChatModel``: no LLM calls, fixed output."""

    def __init__(self, output: LLMClassificationOutput) -> None:
        self.output = output

    def with_structured_output(self, schema: type) -> Runnable[Any, Any]:
        return RunnableLambda(lambda _: self.output)


@pytest.mark.unit
def test_classify_returns_the_structured_output() -> None:
    fake_output = LLMClassificationOutput(
        clause_type=ClauseType.COVERAGE, confidence=0.87
    )
    fake_llm = cast(BaseChatModel, FakeChatModel(fake_output))
    classifier = LangchainClauseClassifier(fake_llm)

    result = classifier.classify("Título", "Texto da cláusula.")

    assert result == (ClauseType.COVERAGE, 0.87)


@pytest.mark.unit
def test_classify_passes_title_and_text_into_the_prompt() -> None:
    seen: dict[str, str] = {}

    class RecordingChatModel:
        def with_structured_output(self, schema: type) -> Runnable[Any, Any]:
            def record(prompt_value: Any) -> LLMClassificationOutput:
                seen["rendered"] = prompt_value.to_string()
                return LLMClassificationOutput(
                    clause_type=ClauseType.EXCLUSION, confidence=0.5
                )

            return RunnableLambda(record)

    fake_llm = cast(BaseChatModel, RecordingChatModel())
    classifier = LangchainClauseClassifier(fake_llm)

    classifier.classify("Riscos Excluídos", "Não estão cobertos os seguintes riscos.")

    assert "Riscos Excluídos" in seen["rendered"]
    assert "Não estão cobertos os seguintes riscos." in seen["rendered"]
