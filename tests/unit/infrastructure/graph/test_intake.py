"""Unit tests for the intake node ([M4-02]).

A fake ``BaseChatModel`` and a stub retriever -- no network, no compiled graph
except the one test that proves the ``Runtime[GraphContext]`` injection path.
Extraction accuracy against the synthetic claims is a separate ``eval``-marked
measurement (``scripts/eval_intake.py`` / ``tests/eval``).
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
from infrastructure.evaluation.synthetic_claims_schema import MissingFactType
from infrastructure.graph.context import GraphContext, RetrievalPort
from infrastructure.graph.nodes.intake import _invoke_with_retry, intake
from infrastructure.graph.prompts.scope_preamble import SCOPE_PREAMBLE
from infrastructure.graph.schemas import IntakeOutput, MissingInfoTag
from infrastructure.graph.state import ExtractedEntities


class _FakeRaw:
    """Stand-in for the AIMessage returned alongside a structured parse."""

    def __init__(self, usage_metadata: dict[str, int] | None) -> None:
        self.usage_metadata = usage_metadata


class FakeChatModel:
    """Stand-in for a ``BaseChatModel`` supporting ``with_structured_output``.

    ``fail_times`` transient failures are raised before the parsed output is
    returned, to exercise the node's retry path.
    """

    def __init__(
        self,
        parsed: IntakeOutput,
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
    """A ``RetrievalPort`` intake never calls but ``GraphContext`` requires."""

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


def _run(
    model: FakeChatModel, raw_claim_text: str = "bati o carro no portao"
) -> dict[str, object]:
    state = {"claim_id": "c1", "raw_claim_text": raw_claim_text}
    return intake(cast(Any, state), Runtime(context=_context(model)))


_FULL_OUTPUT = IntakeOutput(
    event_type="colisão",
    event_date="faz umas duas semanas",
    description="bateu contra uma mureta em dia de chuva",
    estimated_amount=15000.0,
    vehicle_info="carro de passeio, uso particular",
    susep_process=None,
    product_line="CASCO",
    missing_information=[],
)


@pytest.mark.unit
def test_intake_maps_structured_output_onto_extracted_entities() -> None:
    result = _run(FakeChatModel(_FULL_OUTPUT))

    assert result["entities"] == ExtractedEntities(
        event_type="colisão",
        event_date="faz umas duas semanas",
        description="bateu contra uma mureta em dia de chuva",
        estimated_amount=15000.0,
        vehicle_info="carro de passeio, uso particular",
        susep_process=None,
        product_line="CASCO",
    )
    assert result["missing_information"] == []


@pytest.mark.unit
def test_intake_keeps_every_field_none_when_the_model_extracts_nothing() -> None:
    result = _run(FakeChatModel(IntakeOutput()))

    assert result["entities"] == ExtractedEntities()


@pytest.mark.unit
def test_intake_routes_missing_information_tags_to_state_and_dedupes() -> None:
    output = IntakeOutput(
        product_line="CASCO",
        missing_information=[
            "data_evento_vigencia",
            "valor_franquia_limite",
            "data_evento_vigencia",
        ],
    )

    result = _run(FakeChatModel(output))

    assert result["missing_information"] == [
        "data_evento_vigencia",
        "valor_franquia_limite",
    ]


@pytest.mark.unit
def test_intake_does_not_invent_a_susep_process_when_the_model_returns_none() -> None:
    result = _run(FakeChatModel(IntakeOutput(product_line="RCF-A")))

    entities = cast(ExtractedEntities, result["entities"])
    assert entities.susep_process is None


@pytest.mark.unit
def test_intake_records_one_audit_event_with_model_and_token_usage() -> None:
    result = _run(
        FakeChatModel(
            _FULL_OUTPUT,
            usage_metadata={
                "input_tokens": 800,
                "output_tokens": 60,
                "total_tokens": 860,
            },
        )
    )

    trail = cast(list[Any], result["audit_trail"])
    assert len(trail) == 1
    event = trail[0]
    assert event.node == "intake"
    assert event.action == "extract_entities"
    assert event.model == "fake-fast-model"
    assert event.token_usage is not None
    assert event.token_usage.input_tokens == 800
    assert event.token_usage.output_tokens == 60
    assert event.token_usage.total_tokens == 860
    assert event.confidence is None


@pytest.mark.unit
def test_intake_audit_event_has_no_token_usage_when_the_model_reports_none() -> None:
    result = _run(FakeChatModel(_FULL_OUTPUT, usage_metadata=None))

    trail = cast(list[Any], result["audit_trail"])
    assert trail[0].token_usage is None


@pytest.mark.unit
def test_intake_prompt_carries_the_scope_preamble_and_every_product_line() -> None:
    model = FakeChatModel(_FULL_OUTPUT)

    _run(model)

    messages = cast(list[BaseMessage], model.received[0])
    system_text = str(messages[0].content)
    assert SCOPE_PREAMBLE in system_text
    for code in ("CASCO", "RCF-A", "ASSIST", "GAR.EST", "CARTA VERDE"):
        assert code in system_text
    assert messages[1].content == "bati o carro no portao"


@pytest.mark.unit
def test_intake_retries_a_transient_failure_then_succeeds() -> None:
    model = FakeChatModel(_FULL_OUTPUT, fail_times=1)
    sleeps: list[float] = []

    result = _invoke_with_retry(
        model.with_structured_output(IntakeOutput, include_raw=True),
        [],
        sleep=sleeps.append,
    )

    assert model.calls == 2
    assert sleeps == [5.0]
    assert result["parsed"] == _FULL_OUTPUT


@pytest.mark.unit
def test_intake_reraises_after_exhausting_retries() -> None:
    def always_fail(_: object) -> dict[str, object]:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        _invoke_with_retry(RunnableLambda(always_fail), [], sleep=lambda _s: None)


@pytest.mark.unit
def test_missing_info_tags_match_the_synthetic_claim_missing_fact_types() -> None:
    assert set(get_args(MissingInfoTag)) == {member.value for member in MissingFactType}


@pytest.mark.unit
def test_intake_runs_as_a_node_in_a_compiled_state_graph() -> None:
    from infrastructure.graph.state import ClaimState

    builder = StateGraph(ClaimState, context_schema=GraphContext)
    builder.add_node("intake", intake)
    builder.add_edge(START, "intake")
    builder.add_edge("intake", END)
    compiled = builder.compile()

    out = compiled.invoke(
        {"claim_id": "c1", "raw_claim_text": "bati o carro"},
        context=_context(FakeChatModel(_FULL_OUTPUT)),
    )

    assert out["entities"].product_line == "CASCO"
    assert [event.node for event in out["audit_trail"]] == ["intake"]
