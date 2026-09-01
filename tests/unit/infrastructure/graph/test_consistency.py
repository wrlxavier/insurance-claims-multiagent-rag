"""Unit tests for the consistency node ([M4-06]).

A fake ``BaseChatModel`` for the semantic leg and literal entities in state --
no network, no compiled graph except the one injection-path test. Signal
accuracy against the synthetic claims is a separate ``eval``-marked measurement
(``scripts/eval_consistency.py`` / ``tests/eval``).
"""

from typing import Any, cast, get_args

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable, RunnableLambda
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from infrastructure.config.enums import LlmProvider
from infrastructure.config.settings import LlmSettings
from infrastructure.graph.context import GraphContext, RetrievalPort
from infrastructure.graph.nodes.consistency import _invoke_with_retry, consistency
from infrastructure.graph.prompts.scope_preamble import SCOPE_PREAMBLE
from infrastructure.graph.schemas import (
    ConsistencyOutput,
    ConsistencySignalItem,
)
from infrastructure.graph.state import (
    ClaimState,
    ConsistencyReport,
    ConsistencySignal,
    ExtractedEntities,
)
from infrastructure.rag.retrieved_clause import RetrievedClause


class _FakeRaw:
    def __init__(self, usage_metadata: dict[str, int] | None) -> None:
        self.usage_metadata = usage_metadata


class FakeChatModel:
    """A ``BaseChatModel`` stand-in serving one ``ConsistencyOutput`` per call."""

    def __init__(
        self,
        responses: ConsistencyOutput | list[ConsistencyOutput],
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
    """A ``RetrievalPort`` the consistency node never calls."""

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


def _coherent() -> ConsistencyOutput:
    return ConsistencyOutput(signals=[])


def _one_llm_signal() -> ConsistencyOutput:
    return ConsistencyOutput(
        signals=[
            ConsistencySignalItem(
                check="narrative_coherence",
                severity="attention",
                detail="O relato diz que o carro estava parado e em movimento.",
            )
        ]
    )


def _run(
    model: FakeChatModel,
    *,
    entities: ExtractedEntities | None = None,
    missing_information: list[str] | None = None,
    raw_claim_text: str = "bati o carro no portao ontem",
) -> dict[str, object]:
    state: dict[str, object] = {"claim_id": "c1", "raw_claim_text": raw_claim_text}
    if entities is not None:
        state["entities"] = entities
    if missing_information is not None:
        state["missing_information"] = missing_information
    return consistency(cast(ClaimState, state), Runtime(context=_context(model)))


# --- merging the two legs -------------------------------------------------


@pytest.mark.unit
def test_merges_deterministic_and_llm_signals_with_the_right_source() -> None:
    entities = ExtractedEntities(
        product_line="GAR.EST", event_type="colisão", description="bati o carro"
    )
    out = _run(FakeChatModel(_one_llm_signal()), entities=entities)

    report = cast(ConsistencyReport, out["consistency"])
    by_source = {s.source for s in report.signals}
    assert by_source == {"deterministic", "llm"}
    det = [s for s in report.signals if s.source == "deterministic"]
    assert det[0].check == "product_line_contradicts_event"
    llm = [s for s in report.signals if s.source == "llm"]
    assert llm[0].check == "narrative_coherence"


@pytest.mark.unit
def test_a_coherent_claim_yields_an_empty_report_and_no_verdict() -> None:
    out = _run(FakeChatModel(_coherent()))

    assert set(out) == {"consistency", "audit_trail"}
    report = cast(ConsistencyReport, out["consistency"])
    assert report.signals == []
    assert not hasattr(report, "verdict")


@pytest.mark.unit
def test_deterministic_signals_survive_an_llm_failure() -> None:
    entities = ExtractedEntities(estimated_amount=-5.0)
    model = FakeChatModel(_coherent(), fail_times=99)

    out = _run(model, entities=entities)

    report = cast(ConsistencyReport, out["consistency"])
    assert [s.check for s in report.signals] == ["amount_non_positive"]
    trail = cast(list[Any], out["audit_trail"])
    semantic = trail[1]
    assert semantic.action == "semantic_judgement"
    assert semantic.model is None
    assert "llm_failed=True" in semantic.node_input


# --- audit trail --------------------------------------------------------


@pytest.mark.unit
def test_records_two_audit_events_one_per_leg() -> None:
    model = FakeChatModel(
        _one_llm_signal(),
        usage_metadata={
            "input_tokens": 300,
            "output_tokens": 40,
            "total_tokens": 340,
        },
    )
    out = _run(model, entities=ExtractedEntities(estimated_amount=-1.0))

    trail = cast(list[Any], out["audit_trail"])
    assert [(e.node, e.action) for e in trail] == [
        ("consistency", "deterministic_checks"),
        ("consistency", "semantic_judgement"),
    ]
    deterministic, semantic = trail
    assert deterministic.model is None
    assert deterministic.token_usage is None
    assert deterministic.confidence is None
    assert "amount=1" in deterministic.node_input
    assert semantic.model == "fake-fast-model"
    assert semantic.token_usage is not None
    assert semantic.token_usage.total_tokens == 340
    assert semantic.confidence is None


@pytest.mark.unit
def test_semantic_audit_has_no_token_usage_when_the_model_reports_none() -> None:
    out = _run(FakeChatModel(_coherent()))
    trail = cast(list[Any], out["audit_trail"])
    assert trail[1].token_usage is None


# --- prompt ------------------------------------------------------------


@pytest.mark.unit
def test_prompt_carries_the_scope_preamble_and_the_no_verdict_instruction() -> None:
    model = FakeChatModel(_coherent())

    _run(model, entities=ExtractedEntities(event_type="colisão"))

    messages = cast(list[BaseMessage], model.received[0])
    system_text = str(messages[0].content)
    assert SCOPE_PREAMBLE in system_text
    assert "NO verdict" in system_text
    assert "fraud" in system_text.lower()
    assert messages[1].content == "bati o carro no portao ontem"


# --- retry ------------------------------------------------------------


@pytest.mark.unit
def test_transient_failure_is_retried_then_succeeds() -> None:
    model = FakeChatModel(_coherent(), fail_times=1)
    sleeps: list[float] = []

    result = _invoke_with_retry(
        model.with_structured_output(ConsistencyOutput, include_raw=True),
        [],
        sleep=sleeps.append,
    )

    assert model.calls == 2
    assert sleeps == [5.0]
    assert result["parsed"] == _coherent()


@pytest.mark.unit
def test_transient_failure_propagates_after_exhausting_retries() -> None:
    def always_fail(_: object) -> dict[str, object]:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        _invoke_with_retry(RunnableLambda(always_fail), [], sleep=lambda _s: None)


# --- graph wiring + vocabulary guards ---------------------------------


@pytest.mark.unit
def test_runs_as_a_node_in_a_compiled_state_graph() -> None:
    builder: Any = StateGraph(ClaimState, context_schema=GraphContext)
    builder.add_node("consistency", consistency)
    builder.add_edge(START, "consistency")
    builder.add_edge("consistency", END)
    compiled = builder.compile()

    out = compiled.invoke(
        {
            "claim_id": "c1",
            "raw_claim_text": "bati o carro",
            "entities": ExtractedEntities(estimated_amount=-2.0),
        },
        context=_context(FakeChatModel(_one_llm_signal())),
    )

    assert isinstance(out["consistency"], ConsistencyReport)
    assert [event.node for event in out["audit_trail"]] == [
        "consistency",
        "consistency",
    ]


@pytest.mark.unit
def test_consistency_output_carries_no_verdict_field() -> None:
    assert "verdict" not in ConsistencyOutput.model_fields
    assert "verdict" not in ConsistencySignalItem.model_fields


@pytest.mark.unit
def test_llm_signal_literals_are_a_subset_of_the_state_signal_literals() -> None:
    item_severity = set(
        get_args(ConsistencySignalItem.model_fields["severity"].annotation)
    )
    state_severity = set(
        get_args(ConsistencySignal.model_fields["severity"].annotation)
    )
    assert item_severity == state_severity
    assert item_severity == {"info", "attention"}
