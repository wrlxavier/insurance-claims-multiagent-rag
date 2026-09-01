"""Unit tests for the compatibility assessment node ([M4-05]).

A fake ``BaseChatModel`` and literal ``Citation`` objects in state -- no
network, no compiled graph except the one injection-path test. Verdict accuracy
against the golden set is a separate ``eval``-marked measurement
(``scripts/eval_compatibility.py`` / ``tests/eval``).
"""

from typing import Any, cast, get_args

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable, RunnableLambda
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from domain.clause_classification import ClauseType
from domain.verdict import Verdict
from infrastructure.config.enums import LlmProvider
from infrastructure.config.settings import LlmSettings
from infrastructure.graph.context import GraphContext, RetrievalPort
from infrastructure.graph.nodes.compatibility import (
    MAX_GROUNDING_ATTEMPTS,
    _invoke_with_retry,
    compatibility,
)
from infrastructure.graph.prompts.scope_preamble import SCOPE_PREAMBLE
from infrastructure.graph.schemas import (
    CompatibilityOutput,
    CompatibilityVerdict,
    ReasonedAssertion,
)
from infrastructure.graph.state import Citation, ClaimState, CompatibilityAssessment
from infrastructure.rag.retrieved_clause import RetrievedClause


class _FakeRaw:
    def __init__(self, usage_metadata: dict[str, int] | None) -> None:
        self.usage_metadata = usage_metadata


class FakeChatModel:
    """A ``BaseChatModel`` stand-in serving a queue of structured responses.

    ``responses`` is one ``CompatibilityOutput`` or a list of them -- call N
    returns element N, and the last element repeats. ``fail_times`` transient
    failures are raised first, to exercise the retry path.
    """

    def __init__(
        self,
        responses: CompatibilityOutput | list[CompatibilityOutput],
        *,
        usage_metadata: dict[str, int] | None = None,
        fail_times: int = 0,
    ) -> None:
        self._responses = responses if isinstance(responses, list) else [responses]
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
            index = min(self.calls - self.fail_times - 1, len(self._responses) - 1)
            return {
                "parsed": self._responses[index],
                "raw": _FakeRaw(self.usage_metadata),
            }

        return RunnableLambda(_invoke)


class _StubRetriever:
    """A ``RetrievalPort`` the compatibility node never calls."""

    def retrieve(
        self, question: str, *, k: int, metadata_filter: object | None = None
    ) -> list[RetrievedClause]:
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


def _citation(
    clause_id: str, *, clause_type: ClauseType = ClauseType.COVERAGE
) -> Citation:
    return Citation(
        clause_id=clause_id,
        document_id="doc-1",
        susep_process="15414.900000/2013-00",
        clause_type=clause_type,
        relevance_score=0.9,
        excerpt=f"Texto da {clause_id}.",
    )


def _run(
    model: FakeChatModel,
    *,
    citations: list[Citation] | None,
    raw_claim_text: str = "bati o carro no portao ontem",
) -> dict[str, object]:
    state: dict[str, object] = {"claim_id": "c1", "raw_claim_text": raw_claim_text}
    if citations is not None:
        state["citations"] = citations
    return compatibility(cast(ClaimState, state), Runtime(context=_context(model)))


def _grounded_output(verdict: str = "incompatible") -> CompatibilityOutput:
    return CompatibilityOutput(
        verdict=cast(CompatibilityVerdict, verdict),
        assertions=[
            ReasonedAssertion(
                statement="A colisão está entre os eventos cobertos.",
                clause_ids=["doc-1:1.1"],
            ),
            ReasonedAssertion(
                statement="Mas a exclusão de uso remunerado afasta a cobertura.",
                clause_ids=["doc-1:2.4"],
            ),
        ],
        confidence=0.82,
    )


_TWO_CLAUSES = [
    _citation("doc-1:1.1"),
    _citation("doc-1:2.4", clause_type=ClauseType.EXCLUSION),
]


# --- mapping the structured output onto state -------------------------------


@pytest.mark.unit
def test_maps_structured_output_onto_the_compatibility_assessment() -> None:
    out = _run(FakeChatModel(_grounded_output()), citations=_TWO_CLAUSES)

    assessment = cast(CompatibilityAssessment, out["compatibility"])
    assert assessment.verdict is Verdict.INCOMPATIBLE
    assert "A colisão está entre os eventos cobertos." in assessment.reasoning
    assert "[cláusulas: doc-1:2.4]" in assessment.reasoning
    assert [c.clause_id for c in assessment.citations] == ["doc-1:1.1", "doc-1:2.4"]
    assert assessment.confidence == pytest.approx(0.82)


@pytest.mark.unit
def test_hydrates_only_the_clauses_the_assertions_cite() -> None:
    citations = [
        _citation("doc-1:1.1"),
        _citation("doc-1:2.4", clause_type=ClauseType.EXCLUSION),
        _citation("doc-1:9.9"),
    ]
    output = CompatibilityOutput(
        verdict="compatible",
        assertions=[
            ReasonedAssertion(statement="Coberto.", clause_ids=["doc-1:1.1"]),
            ReasonedAssertion(statement="Confirma.", clause_ids=["doc-1:9.9"]),
        ],
        confidence=0.7,
    )

    out = _run(FakeChatModel(output), citations=citations)

    assessment = cast(CompatibilityAssessment, out["compatibility"])
    assert [c.clause_id for c in assessment.citations] == ["doc-1:1.1", "doc-1:9.9"]


# --- the grounding-retry loop ---------------------------------------------


@pytest.mark.unit
def test_retries_when_an_assertion_cites_a_clause_that_was_not_retrieved() -> None:
    ungrounded = CompatibilityOutput(
        verdict="incompatible",
        assertions=[
            ReasonedAssertion(statement="Exclusão.", clause_ids=["doc-1:ghost"])
        ],
        confidence=0.6,
    )
    model = FakeChatModel([ungrounded, _grounded_output()])

    out = _run(model, citations=_TWO_CLAUSES)

    assert model.calls == 2
    assessment = cast(CompatibilityAssessment, out["compatibility"])
    assert assessment.verdict is Verdict.INCOMPATIBLE
    # the corrective turn was appended to the conversation
    assert len(model.received[1]) == len(model.received[0]) + 1


@pytest.mark.unit
def test_degrades_to_insufficient_information_after_exhausting_grounding_retries() -> (
    None
):
    ungrounded = CompatibilityOutput(
        verdict="compatible",
        assertions=[ReasonedAssertion(statement="Coberto.", clause_ids=[])],
        confidence=0.9,
    )
    model = FakeChatModel(ungrounded)

    out = _run(model, citations=_TWO_CLAUSES)

    assert model.calls == MAX_GROUNDING_ATTEMPTS
    assessment = cast(CompatibilityAssessment, out["compatibility"])
    assert assessment.verdict is Verdict.INSUFFICIENT_INFORMATION
    assert assessment.citations == []
    assert assessment.confidence == 0.0
    assert "não pôde ser fundamentada" in assessment.reasoning

    trail = cast(list[Any], out["audit_trail"])
    assert trail[0].node == "compatibility"
    assert trail[0].model == "fake-reasoning-model"
    assert trail[0].confidence == 0.0


@pytest.mark.unit
def test_insufficient_information_verdict_is_accepted_without_assertions() -> None:
    output = CompatibilityOutput(
        verdict="insufficient_information", assertions=[], confidence=0.2
    )
    model = FakeChatModel(output)

    out = _run(model, citations=_TWO_CLAUSES)

    assert model.calls == 1
    assessment = cast(CompatibilityAssessment, out["compatibility"])
    assert assessment.verdict is Verdict.INSUFFICIENT_INFORMATION
    assert assessment.reasoning == "Nenhuma afirmação fundamentada foi produzida."


# --- the no-context guard -------------------------------------------------


@pytest.mark.unit
def test_abstains_without_calling_the_model_when_no_clauses_were_retrieved() -> None:
    model = FakeChatModel(_grounded_output())

    out = _run(model, citations=[])

    assert model.calls == 0
    assessment = cast(CompatibilityAssessment, out["compatibility"])
    assert assessment.verdict is Verdict.INSUFFICIENT_INFORMATION
    assert assessment.citations == []

    trail = cast(list[Any], out["audit_trail"])
    assert trail[0].node == "compatibility"
    assert trail[0].action == "assess"
    assert trail[0].model is None
    assert trail[0].token_usage is None
    assert trail[0].confidence == 0.0


@pytest.mark.unit
def test_missing_citations_key_is_treated_as_no_context() -> None:
    model = FakeChatModel(_grounded_output())

    out = _run(model, citations=None)

    assert model.calls == 0
    assert (
        cast(CompatibilityAssessment, out["compatibility"]).verdict
        is Verdict.INSUFFICIENT_INFORMATION
    )


# --- prompt + audit -----------------------------------------------------


@pytest.mark.unit
def test_prompt_carries_the_scope_preamble_the_clause_list_and_the_weighing_rule() -> (
    None
):
    model = FakeChatModel(_grounded_output())

    _run(model, citations=_TWO_CLAUSES)

    messages = cast(list[BaseMessage], model.received[0])
    system_text = str(messages[0].content)
    assert SCOPE_PREAMBLE in system_text
    assert "doc-1:1.1" in system_text and "doc-1:2.4" in system_text
    assert "exclusion" in system_text.lower()
    assert "weigh" in system_text.lower()
    assert messages[1].content == "bati o carro no portao ontem"


@pytest.mark.unit
def test_records_one_audit_event_with_the_reasoning_model_and_confidence() -> None:
    model = FakeChatModel(
        _grounded_output(),
        usage_metadata={
            "input_tokens": 1200,
            "output_tokens": 90,
            "total_tokens": 1290,
        },
    )

    out = _run(model, citations=_TWO_CLAUSES)

    trail = cast(list[Any], out["audit_trail"])
    assert len(trail) == 1
    event = trail[0]
    assert event.node == "compatibility"
    assert event.action == "assess"
    assert event.model == "fake-reasoning-model"
    assert event.token_usage is not None
    assert event.token_usage.total_tokens == 1290
    assert event.confidence == pytest.approx(0.82)


# --- retry + graph wiring ------------------------------------------------


@pytest.mark.unit
def test_transient_failure_is_retried_then_succeeds() -> None:
    model = FakeChatModel(_grounded_output(), fail_times=1)
    sleeps: list[float] = []

    result = _invoke_with_retry(
        model.with_structured_output(CompatibilityOutput, include_raw=True),
        [],
        sleep=sleeps.append,
    )

    assert model.calls == 2
    assert sleeps == [5.0]
    assert result["parsed"] == _grounded_output()


@pytest.mark.unit
def test_transient_failure_propagates_after_exhausting_retries() -> None:
    def always_fail(_: object) -> dict[str, object]:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        _invoke_with_retry(RunnableLambda(always_fail), [], sleep=lambda _s: None)


@pytest.mark.unit
def test_runs_as_a_node_in_a_compiled_state_graph() -> None:
    builder: Any = StateGraph(ClaimState, context_schema=GraphContext)
    builder.add_node("compatibility", compatibility)
    builder.add_edge(START, "compatibility")
    builder.add_edge("compatibility", END)
    compiled = builder.compile()

    out = compiled.invoke(
        {
            "claim_id": "c1",
            "raw_claim_text": "bati o carro",
            "citations": _TWO_CLAUSES,
        },
        context=_context(FakeChatModel(_grounded_output())),
    )

    assert out["compatibility"].verdict is Verdict.INCOMPATIBLE
    assert [event.node for event in out["audit_trail"]] == ["compatibility"]


@pytest.mark.unit
def test_compatibility_verdict_literal_matches_the_domain_vocabulary() -> None:
    assert set(get_args(CompatibilityVerdict)) == {member.value for member in Verdict}
