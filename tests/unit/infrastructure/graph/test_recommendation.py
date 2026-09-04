"""Unit tests for the recommendation node ([M4-08]).

A fake ``BaseChatModel`` for the justification leg and literal upstream results
in state -- no network, one compiled-graph injection test. The two DoD
guarantees are checked as structural properties: the node never emits a citation
no upstream node produced, and an ``insufficient_information`` effective verdict
never carries a confident recommendation. End-to-end accuracy is a separate
``eval``-marked measurement (``scripts/eval_recommendation.py`` / ``tests/eval``).
"""

from typing import Any, cast

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
from infrastructure.graph.nodes.recommendation import (
    _ATTENTION_FLAG_CONFIDENCE_CEILING,
    _INSUFFICIENT_CONFIDENCE_CEILING,
    _invoke_with_retry,
    recommendation,
)
from infrastructure.graph.prompts.recommendation import build_recommendation_prompt
from infrastructure.graph.prompts.scope_preamble import SCOPE_PREAMBLE
from infrastructure.graph.prompts.untrusted_content import wrap_untrusted
from infrastructure.graph.schemas import RecommendationOutput
from infrastructure.graph.state import (
    Citation,
    ClaimState,
    CompatibilityAssessment,
    ConsistencyReport,
    ConsistencySignal,
    Recommendation,
)
from infrastructure.rag.retrieved_clause import RetrievedClause


class _FakeRaw:
    def __init__(self, usage_metadata: dict[str, int] | None) -> None:
        self.usage_metadata = usage_metadata


class FakeChatModel:
    """A ``BaseChatModel`` stand-in serving one ``RecommendationOutput`` per call."""

    def __init__(
        self,
        responses: RecommendationOutput | list[RecommendationOutput],
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
    """A ``RetrievalPort`` the recommendation node never calls."""

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


def _citation(clause_id: str, *, document_id: str = "doc-1") -> Citation:
    return Citation(
        clause_id=clause_id,
        document_id=document_id,
        susep_process="15414.900000/2013-00",
        clause_type=ClauseType.COVERAGE,
        relevance_score=0.9,
        excerpt=f"Texto da cláusula {clause_id}.",
    )


def _assessment(
    *,
    verdict: Verdict = Verdict.COMPATIBLE,
    citations: list[Citation] | None = None,
    confidence: float = 0.8,
    reasoning: str = "1. O evento se enquadra na cobertura. [cláusulas: doc-1:1.1]",
) -> CompatibilityAssessment:
    return CompatibilityAssessment(
        verdict=verdict,
        reasoning=reasoning,
        citations=[_citation("doc-1:1.1")] if citations is None else citations,
        confidence=confidence,
    )


def _signal(
    *, severity: str = "attention", check: str = "narrative_coherence"
) -> ConsistencySignal:
    return ConsistencySignal(
        check=check,
        severity=cast(Any, severity),
        detail="O relato diz que o carro estava parado e em movimento.",
        source="llm",
    )


def _draft(text: str = "Resumo consolidado para o revisor.") -> RecommendationOutput:
    return RecommendationOutput(justification=text)


def _run(
    model: FakeChatModel,
    *,
    compatibility: CompatibilityAssessment | None = None,
    consistency: ConsistencyReport | None = None,
    citations: list[Citation] | None = None,
    context_sufficient: bool | None = None,
    clarification_exhausted: bool | None = None,
    missing_information: list[str] | None = None,
    raw_claim_text: str = "bati o carro no portao ontem",
) -> dict[str, object]:
    state: dict[str, object] = {"claim_id": "c1", "raw_claim_text": raw_claim_text}
    if compatibility is not None:
        state["compatibility"] = compatibility
    if consistency is not None:
        state["consistency"] = consistency
    if citations is not None:
        state["citations"] = citations
    if context_sufficient is not None:
        state["context_sufficient"] = context_sufficient
    if clarification_exhausted is not None:
        state["clarification_exhausted"] = clarification_exhausted
    if missing_information is not None:
        state["missing_information"] = missing_information
    return recommendation(cast(ClaimState, state), Runtime(context=_context(model)))


# --- consolidating an assessment --------------------------------------------


@pytest.mark.unit
def test_consolidates_a_compatible_assessment() -> None:
    model = FakeChatModel(
        _draft("O evento é compatível; sustentado por doc-1:1.1."),
        usage_metadata={"input_tokens": 200, "output_tokens": 30, "total_tokens": 230},
    )
    out = _run(
        model,
        compatibility=_assessment(verdict=Verdict.COMPATIBLE, confidence=0.8),
        consistency=ConsistencyReport(signals=[]),
    )

    rec = cast(Recommendation, out["recommendation"])
    assert "revisão humana" in rec.recommended_action
    assert rec.justification == "O evento é compatível; sustentado por doc-1:1.1."
    assert [c.clause_id for c in rec.citations] == ["doc-1:1.1"]
    assert rec.consistency_flags == []
    assert rec.confidence == 0.8

    trail = cast(list[Any], out["audit_trail"])
    assert len(trail) == 1
    assert (trail[0].node, trail[0].action) == ("recommendation", "consolidate")
    assert trail[0].model == "fake-fast-model"
    assert trail[0].token_usage is not None
    assert trail[0].token_usage.total_tokens == 230
    assert trail[0].confidence == 0.8


@pytest.mark.unit
def test_incompatible_assessment_maps_to_the_priority_review_action() -> None:
    out = _run(
        FakeChatModel(_draft()),
        compatibility=_assessment(verdict=Verdict.INCOMPATIBLE, confidence=0.7),
        consistency=ConsistencyReport(signals=[]),
    )
    rec = cast(Recommendation, out["recommendation"])
    assert "prioridade" in rec.recommended_action
    assert rec.confidence == 0.7


@pytest.mark.unit
def test_only_the_changed_state_keys_are_returned() -> None:
    out = _run(
        FakeChatModel(_draft()),
        compatibility=_assessment(),
        consistency=ConsistencyReport(signals=[]),
    )
    assert set(out) == {"recommendation", "audit_trail"}


# --- citation grounding (DoD) ----------------------------------------------


@pytest.mark.unit
def test_citations_are_exactly_the_grounded_upstream_subset() -> None:
    grounded = [_citation("doc-1:1.1")]
    out = _run(
        FakeChatModel(_draft()),
        compatibility=_assessment(citations=grounded),
        citations=[_citation("doc-1:1.1"), _citation("doc-1:9.9")],
        consistency=ConsistencyReport(signals=[]),
    )
    rec = cast(Recommendation, out["recommendation"])
    assert [c.clause_id for c in rec.citations] == ["doc-1:1.1"]


@pytest.mark.unit
def test_never_emits_a_citation_no_upstream_node_produced() -> None:
    # The model returns junk that names a clause id; it has no citation field,
    # so it cannot land in the recommendation. Across every path the emitted
    # citations stay a subset of what compatibility (hence retrieval) produced.
    junk = _draft("Cito a cláusula inventada doc-9:ghost e doc-1:1.1.")
    retrieval_ids = {"doc-1:1.1", "doc-1:2.2"}
    for compat in (
        _assessment(citations=[_citation("doc-1:1.1")]),
        _assessment(citations=[_citation("doc-1:1.1"), _citation("doc-1:2.2")]),
        _assessment(verdict=Verdict.INSUFFICIENT_INFORMATION, citations=[]),
        None,
    ):
        out = _run(
            FakeChatModel(junk),
            compatibility=compat,
            citations=[_citation("doc-1:1.1"), _citation("doc-1:2.2")],
            context_sufficient=None if compat is not None else False,
            consistency=ConsistencyReport(signals=[]),
        )
        rec = cast(Recommendation, out["recommendation"])
        emitted = {c.clause_id for c in rec.citations}
        upstream = (
            {c.clause_id for c in compat.citations} if compat is not None else set()
        )
        assert emitted <= upstream
        assert emitted <= retrieval_ids


@pytest.mark.unit
def test_recommendation_output_carries_only_the_justification_field() -> None:
    assert set(RecommendationOutput.model_fields) == {"justification"}


# --- an insufficient upstream verdict cannot become confident (DoD) --------


@pytest.mark.unit
def test_insufficient_compatibility_verdict_stays_unconfident() -> None:
    out = _run(
        FakeChatModel(_draft()),
        compatibility=_assessment(
            verdict=Verdict.INSUFFICIENT_INFORMATION, citations=[], confidence=0.95
        ),
        consistency=ConsistencyReport(signals=[]),
    )
    rec = cast(Recommendation, out["recommendation"])
    assert rec.confidence <= _INSUFFICIENT_CONFIDENCE_CEILING
    assert "inconclusiva" in rec.recommended_action


@pytest.mark.unit
def test_clarification_exhausted_is_low_confidence_and_skips_the_model() -> None:
    model = FakeChatModel(_draft())
    out = _run(
        model,
        clarification_exhausted=True,
        missing_information=["data_evento_vigencia", "uso_do_veiculo"],
    )
    rec = cast(Recommendation, out["recommendation"])
    assert model.calls == 0
    assert rec.confidence <= _INSUFFICIENT_CONFIDENCE_CEILING
    assert rec.citations == []
    assert "data_evento_vigencia" in rec.recommended_action
    assert "uso_do_veiculo" in rec.recommended_action
    trail = cast(list[Any], out["audit_trail"])
    assert trail[0].model is None
    assert "posture=claimant_gaps" in trail[0].node_input


@pytest.mark.unit
def test_insufficient_retrieval_skips_the_model() -> None:
    model = FakeChatModel(_draft())
    out = _run(model, context_sufficient=False, citations=[_citation("doc-1:1.1")])
    rec = cast(Recommendation, out["recommendation"])
    assert model.calls == 0
    assert rec.citations == []
    assert rec.confidence <= _INSUFFICIENT_CONFIDENCE_CEILING
    assert "revisão manual de cláusulas" in rec.recommended_action
    trail = cast(list[Any], out["audit_trail"])
    assert trail[0].model is None
    assert "posture=retrieval_miss" in trail[0].node_input


# --- consistency flags stay separate (DoD) --------------------------------


@pytest.mark.unit
def test_consistency_flags_are_carried_verbatim_and_kept_separate() -> None:
    flag = _signal(severity="attention")
    out = _run(
        FakeChatModel(_draft()),
        compatibility=_assessment(verdict=Verdict.COMPATIBLE, confidence=0.9),
        consistency=ConsistencyReport(signals=[flag]),
    )
    rec = cast(Recommendation, out["recommendation"])
    assert rec.consistency_flags == [flag]
    # the flag does not change the verdict-driven action ...
    assert "revisão humana" in rec.recommended_action
    assert "prioridade" not in rec.recommended_action
    # ... but an unresolved attention flag caps confidence.
    assert rec.confidence == _ATTENTION_FLAG_CONFIDENCE_CEILING


@pytest.mark.unit
def test_info_severity_flags_do_not_cap_confidence() -> None:
    out = _run(
        FakeChatModel(_draft()),
        compatibility=_assessment(verdict=Verdict.COMPATIBLE, confidence=0.9),
        consistency=ConsistencyReport(signals=[_signal(severity="info")]),
    )
    assert cast(Recommendation, out["recommendation"]).confidence == 0.9


# --- degradation ---------------------------------------------------------


@pytest.mark.unit
def test_degrades_to_a_deterministic_justification_when_the_model_fails() -> None:
    model = FakeChatModel(_draft(), fail_times=99)
    out = _run(
        model,
        compatibility=_assessment(
            verdict=Verdict.COMPATIBLE,
            citations=[_citation("doc-1:1.1")],
            reasoning="1. Enquadra-se na cobertura. [cláusulas: doc-1:1.1]",
        ),
        consistency=ConsistencyReport(signals=[_signal()]),
    )
    rec = cast(Recommendation, out["recommendation"])
    assert rec.justification  # non-empty
    assert "doc-1:1.1" in rec.justification
    assert "Pontos de atenção" in rec.justification
    trail = cast(list[Any], out["audit_trail"])
    assert trail[0].model is None
    assert "llm_failed=True" in trail[0].node_input


@pytest.mark.unit
def test_blank_model_justification_falls_back_to_the_template() -> None:
    out = _run(
        FakeChatModel(_draft("   ")),
        compatibility=_assessment(),
        consistency=ConsistencyReport(signals=[]),
    )
    rec = cast(Recommendation, out["recommendation"])
    assert rec.justification.strip()
    assert "Cláusulas consideradas: doc-1:1.1." in rec.justification


@pytest.mark.unit
def test_transient_failure_is_retried_then_succeeds() -> None:
    model = FakeChatModel(_draft(), fail_times=1)
    sleeps: list[float] = []

    result = _invoke_with_retry(
        model.with_structured_output(RecommendationOutput, include_raw=True),
        [],
        sleep=sleeps.append,
    )

    assert model.calls == 2
    assert sleeps == [5.0]
    assert result["parsed"] == _draft()


@pytest.mark.unit
def test_one_transient_failure_still_yields_the_model_justification() -> None:
    model = FakeChatModel(_draft("Do modelo, após uma tentativa."), fail_times=1)
    out = _run(
        model,
        compatibility=_assessment(),
        consistency=ConsistencyReport(signals=[]),
    )
    rec = cast(Recommendation, out["recommendation"])
    assert model.calls == 2
    assert rec.justification == "Do modelo, após uma tentativa."
    assert cast(list[Any], out["audit_trail"])[0].model == "fake-fast-model"


# --- prompt ------------------------------------------------------------


@pytest.mark.unit
def test_prompt_carries_the_scope_preamble_the_clauses_and_the_flags() -> None:
    prompt = build_recommendation_prompt(
        None,
        _assessment(verdict=Verdict.INCOMPATIBLE, citations=[_citation("doc-1:3.3")]),
        [_signal(severity="attention")],
        [_citation("doc-1:3.3")],
    )
    assert SCOPE_PREAMBLE in prompt
    assert "doc-1:3.3" in prompt
    assert "carro estava parado e em movimento" in prompt
    assert "incompatible" in prompt


@pytest.mark.unit
def test_prompt_is_the_system_message_and_the_narrative_is_the_human_message() -> None:
    model = FakeChatModel(_draft())
    _run(
        model,
        compatibility=_assessment(),
        consistency=ConsistencyReport(signals=[]),
        raw_claim_text="bati o carro",
    )
    messages = cast(list[BaseMessage], model.received[0])
    assert SCOPE_PREAMBLE in str(messages[0].content)
    assert messages[1].content == wrap_untrusted("claim_narrative", "bati o carro")


# --- graph wiring ----------------------------------------------------------


@pytest.mark.unit
def test_runs_as_a_node_in_a_compiled_state_graph() -> None:
    builder: Any = StateGraph(ClaimState, context_schema=GraphContext)
    builder.add_node("recommendation", recommendation)
    builder.add_edge(START, "recommendation")
    builder.add_edge("recommendation", END)
    compiled = builder.compile()

    out = compiled.invoke(
        {
            "claim_id": "c1",
            "raw_claim_text": "bati o carro",
            "compatibility": _assessment(),
            "consistency": ConsistencyReport(signals=[]),
        },
        context=_context(FakeChatModel(_draft())),
    )

    assert isinstance(out["recommendation"], Recommendation)
    assert [event.node for event in out["audit_trail"]] == ["recommendation"]
