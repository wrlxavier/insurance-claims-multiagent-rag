"""Unit tests for the clarification node ([M4-03]).

A fake ``BaseChatModel`` and a stub retriever -- no network. The loop's
end-to-end termination is exercised in ``test_claim_graph.py``; this file is
the node in isolation: one question per gap, the template fallback, the round
counter, the audit event.
"""

from typing import Any, cast

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable, RunnableLambda
from langgraph.runtime import Runtime

from infrastructure.config.enums import LlmProvider
from infrastructure.config.settings import LlmSettings
from infrastructure.graph.context import GraphContext, RetrievalPort
from infrastructure.graph.nodes.clarification import clarification
from infrastructure.graph.prompts.clarification import CLARIFICATION_FALLBACK_TEMPLATES
from infrastructure.graph.prompts.scope_preamble import SCOPE_PREAMBLE
from infrastructure.graph.schemas import ClarificationOutput, ClarificationQuestionItem
from infrastructure.graph.state import ClarificationQuestion, ExtractedEntities


class _FakeRaw:
    def __init__(self, usage_metadata: dict[str, int] | None) -> None:
        self.usage_metadata = usage_metadata


class FakeChatModel:
    """Stand-in for a ``BaseChatModel`` supporting ``with_structured_output``."""

    def __init__(
        self,
        parsed: ClarificationOutput,
        *,
        usage_metadata: dict[str, int] | None = None,
        fail_times: int = 0,
    ) -> None:
        self.parsed = parsed
        self.usage_metadata = usage_metadata
        self.fail_times = fail_times
        self.calls = 0
        self.received: list[Any] = []

    def with_structured_output(
        self, schema: type, include_raw: bool = False
    ) -> Runnable[Any, Any]:
        def _invoke(messages: Any) -> dict[str, object]:
            self.calls += 1
            self.received.append(messages)
            if self.calls <= self.fail_times:
                raise RuntimeError("transient provider failure")
            return {"parsed": self.parsed, "raw": _FakeRaw(self.usage_metadata)}

        return RunnableLambda(_invoke)


class _StubRetriever:
    def retrieve(
        self, question: str, *, k: int, metadata_filter: object | None = None
    ) -> list[str]:
        return []


def _llm_settings() -> LlmSettings:
    return LlmSettings(
        LLM_PROVIDER=LlmProvider.OPENAI,
        LLM_API_KEY="test-key",
        LLM_MODEL_FAST="fake-fast-model",
        LLM_MODEL_REASONING="fake-reasoning-model",
        EMBEDDING_MODEL="embed-model",
        RERANKER_MODEL="rerank-model",
        _env_file=None,
    )


def _context(model: FakeChatModel) -> GraphContext:
    chat = cast(BaseChatModel, model)
    return GraphContext(
        fast_model=chat,
        reasoning_model=chat,
        retriever=cast(RetrievalPort, _StubRetriever()),
        llm_settings=_llm_settings(),
    )


def _run(model: FakeChatModel, state: dict[str, object]) -> dict[str, object]:
    base: dict[str, object] = {"claim_id": "c1", "raw_claim_text": "bati o carro"}
    return clarification(cast(Any, {**base, **state}), Runtime(context=_context(model)))


def _output(*tags: str) -> ClarificationOutput:
    return ClarificationOutput(
        questions=[
            ClarificationQuestionItem(field=tag, question=f"Pergunta sobre {tag}?")
            for tag in tags
        ]
    )


@pytest.mark.unit
def test_one_question_per_gap_mapped_onto_state() -> None:
    result = _run(
        FakeChatModel(_output("data_evento_vigencia", "valor_franquia_limite")),
        {"missing_information": ["data_evento_vigencia", "valor_franquia_limite"]},
    )

    questions = cast(list[ClarificationQuestion], result["clarification_questions"])
    assert [q.field for q in questions] == [
        "data_evento_vigencia",
        "valor_franquia_limite",
    ]
    assert all(q.question for q in questions)


@pytest.mark.unit
def test_accumulates_onto_prior_questions() -> None:
    prior = [ClarificationQuestion(field="ambito_geografico", question="Onde foi?")]
    result = _run(
        FakeChatModel(_output("data_evento_vigencia")),
        {
            "missing_information": ["data_evento_vigencia"],
            "clarification_questions": prior,
        },
    )

    questions = cast(list[ClarificationQuestion], result["clarification_questions"])
    assert [q.field for q in questions] == ["ambito_geografico", "data_evento_vigencia"]


@pytest.mark.unit
def test_increments_the_round_counter() -> None:
    absent = _run(
        FakeChatModel(_output("data_evento_vigencia")),
        {"missing_information": ["data_evento_vigencia"]},
    )
    assert absent["clarification_rounds"] == 1

    second = _run(
        FakeChatModel(_output("data_evento_vigencia")),
        {"missing_information": ["data_evento_vigencia"], "clarification_rounds": 1},
    )
    assert second["clarification_rounds"] == 2


@pytest.mark.unit
def test_fills_a_gap_the_model_omitted_from_a_template() -> None:
    result = _run(
        FakeChatModel(_output("data_evento_vigencia")),  # omits the second gap
        {"missing_information": ["data_evento_vigencia", "valor_franquia_limite"]},
    )

    questions = cast(list[ClarificationQuestion], result["clarification_questions"])
    by_tag = {q.field: q.question for q in questions}
    assert (
        by_tag["valor_franquia_limite"]
        == CLARIFICATION_FALLBACK_TEMPLATES["valor_franquia_limite"]
    )


@pytest.mark.unit
def test_falls_back_to_templates_for_every_gap_when_the_llm_fails() -> None:
    model = FakeChatModel(_output("data_evento_vigencia"), fail_times=99)
    result = _run(
        model,
        {"missing_information": ["data_evento_vigencia", "ambito_geografico"]},
    )

    questions = cast(list[ClarificationQuestion], result["clarification_questions"])
    assert {q.field for q in questions} == {"data_evento_vigencia", "ambito_geografico"}
    assert all(
        q.question == CLARIFICATION_FALLBACK_TEMPLATES[q.field] for q in questions
    )
    trail = cast(list[Any], result["audit_trail"])
    assert trail[0].model is None
    assert "llm_failed" in (trail[0].node_input or "")


@pytest.mark.unit
def test_records_one_audit_event_with_model_and_token_usage() -> None:
    result = _run(
        FakeChatModel(
            _output("data_evento_vigencia"),
            usage_metadata={
                "input_tokens": 500,
                "output_tokens": 40,
                "total_tokens": 540,
            },
        ),
        {"missing_information": ["data_evento_vigencia"]},
    )

    trail = cast(list[Any], result["audit_trail"])
    assert len(trail) == 1
    event = trail[0]
    assert event.node == "clarification"
    assert event.action == "generate_questions"
    assert event.model == "fake-fast-model"
    assert event.token_usage is not None
    assert event.token_usage.total_tokens == 540
    assert event.confidence is None


@pytest.mark.unit
def test_prompt_carries_scope_preamble_and_the_known_facts() -> None:
    model = FakeChatModel(_output("valor_franquia_limite"))
    _run(
        model,
        {
            "missing_information": ["valor_franquia_limite"],
            "entities": ExtractedEntities(event_type="colisão", product_line="CASCO"),
        },
    )

    messages = cast(list[BaseMessage], model.received[0])
    system_text = str(messages[0].content)
    assert SCOPE_PREAMBLE in system_text
    assert "valor_franquia_limite" in system_text
    assert "colisão" in system_text
    assert messages[1].content == "bati o carro"
