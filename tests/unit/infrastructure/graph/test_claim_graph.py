"""The compiled claim graph: the loop and the retrieval hand-off ([M4-03]/[M4-04]).

``route_after_intake`` / ``route_after_retrieval`` in isolation, the graph's
structure, and -- the [M4-03] DoD's core check -- that the
``intake ⇄ clarification`` loop terminates for every incomplete claim in the
M2-04 set. Termination is a property of the router plus
``MAX_CLARIFICATION_ROUNDS``, so a fake LLM plus a stub retriever is the right
tool: no network, and no reliance on LangGraph's ``recursion_limit``.
"""

from pathlib import Path
from typing import Any, cast

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable, RunnableLambda
from langgraph.graph import START

from domain.clause_classification import ClauseType
from infrastructure.config.enums import LlmProvider
from infrastructure.config.settings import LlmSettings
from infrastructure.evaluation.synthetic_claims_schema import SyntheticClaim
from infrastructure.graph.build import (
    MAX_CLARIFICATION_ROUNDS,
    build_claim_graph,
    route_after_intake,
    route_after_retrieval,
)
from infrastructure.graph.context import GraphContext, RetrievalPort
from infrastructure.graph.schemas import (
    ClarificationOutput,
    ClarificationQuestionItem,
    IntakeOutput,
)
from infrastructure.rag.retrieved_clause import RetrievedClause

REPO_ROOT = Path(__file__).resolve().parents[4]
CLAIMS_PATH = REPO_ROOT / "data" / "synthetic_claims" / "claims.jsonl"


class _FakeRaw:
    usage_metadata = None


class _LoopModel:
    """Serves both nodes: intake always reports ``missing`` as unresolved."""

    def __init__(self, missing: list[str]) -> None:
        self.missing = missing

    def with_structured_output(
        self, schema: type, include_raw: bool = False
    ) -> Runnable[Any, Any]:
        def _invoke(_messages: Any) -> dict[str, object]:
            if schema is IntakeOutput:
                parsed: object = IntakeOutput(
                    product_line="CASCO", missing_information=list(self.missing)
                )
            else:
                parsed = ClarificationOutput(
                    questions=[
                        ClarificationQuestionItem(field=tag, question=f"Sobre {tag}?")
                        for tag in self.missing
                    ]
                )
            return {"parsed": parsed, "raw": _FakeRaw()}

        return RunnableLambda(_invoke)


_STUB_HIT = RetrievedClause(
    clause_id="doc-1:1.1",
    document_id="doc-1",
    susep_process="15414.900000/2013-00",
    clause_type=ClauseType.COVERAGE,
    excerpt="A seguradora cobre colisao.",
    score=0.95,
)


class _StubRetriever:
    """Returns one high-scoring clause so a complete claim clears the [M3-07] gate."""

    def retrieve(
        self, question: str, *, k: int, metadata_filter: object | None = None
    ) -> list[RetrievedClause]:
        return [_STUB_HIT]


def _context(missing: list[str]) -> GraphContext:
    model = cast(BaseChatModel, _LoopModel(missing))
    return GraphContext(
        fast_model=model,
        reasoning_model=model,
        retriever=cast(RetrievalPort, _StubRetriever()),
        llm_settings=LlmSettings(
            LLM_PROVIDER=LlmProvider.OPENAI,
            LLM_API_KEY="test-key",
            LLM_MODEL_FAST="fake-fast-model",
            LLM_MODEL_REASONING="fake-reasoning-model",
            EMBEDDING_MODEL="embed-model",
            RERANKER_MODEL="rerank-model",
            _env_file=None,
        ),
    )


def _invoke(missing: list[str]) -> dict[str, Any]:
    compiled = build_claim_graph().compile()
    return compiled.invoke(
        {"claim_id": "c1", "raw_claim_text": "bati o carro"},
        context=_context(missing),
    )


# --- route_after_intake -------------------------------------------------


@pytest.mark.unit
def test_route_proceeds_when_nothing_is_missing() -> None:
    assert route_after_intake({"claim_id": "c", "raw_claim_text": "x"}) == "proceed"


@pytest.mark.unit
def test_route_asks_while_rounds_remain() -> None:
    state = {"claim_id": "c", "raw_claim_text": "x", "missing_information": ["x"]}
    assert route_after_intake(cast(Any, state)) == "clarification"


@pytest.mark.unit
def test_route_exhausts_at_the_cap() -> None:
    state = {
        "claim_id": "c",
        "raw_claim_text": "x",
        "missing_information": ["x"],
        "clarification_rounds": MAX_CLARIFICATION_ROUNDS,
    }
    assert route_after_intake(cast(Any, state)) == "exhausted"


# --- route_after_retrieval -------------------------------------------


@pytest.mark.unit
def test_route_after_retrieval_proceeds_when_context_suffices() -> None:
    state = {"claim_id": "c", "raw_claim_text": "x", "context_sufficient": True}
    assert route_after_retrieval(cast(Any, state)) == "assess"


@pytest.mark.unit
def test_route_after_retrieval_flags_insufficient_context() -> None:
    state = {"claim_id": "c", "raw_claim_text": "x", "context_sufficient": False}
    assert route_after_retrieval(cast(Any, state)) == "insufficient"


# --- structure --------------------------------------------------------


@pytest.mark.unit
def test_graph_has_the_expected_nodes_and_entry() -> None:
    graph = build_claim_graph().compile().get_graph()
    assert {
        "intake",
        "clarification",
        "clarification_exhausted",
        "retrieval",
    } <= set(graph.nodes)
    assert any(e.source == START and e.target == "intake" for e in graph.edges)
    assert any(e.source == "intake" and e.target == "retrieval" for e in graph.edges)


# --- the loop --------------------------------------------------------


@pytest.mark.unit
def test_complete_claim_passes_straight_through() -> None:
    out = _invoke(missing=[])

    assert out.get("clarification_exhausted") in (None, False)
    assert not out.get("clarification_questions")
    assert out.get("clarification_rounds", 0) == 0
    assert [e.node for e in out["audit_trail"]] == ["intake", "retrieval"]
    assert out["context_sufficient"] is True
    assert [c.clause_id for c in out["citations"]] == ["doc-1:1.1"]


@pytest.mark.unit
def test_retrieval_with_no_hits_flags_insufficient_context() -> None:
    class _EmptyRetriever:
        def retrieve(
            self, question: str, *, k: int, metadata_filter: object | None = None
        ) -> list[RetrievedClause]:
            return []

    compiled = build_claim_graph().compile()
    context = _context(missing=[])
    context = GraphContext(
        fast_model=context.fast_model,
        reasoning_model=context.reasoning_model,
        retriever=cast(RetrievalPort, _EmptyRetriever()),
        llm_settings=context.llm_settings,
    )
    out = compiled.invoke(
        {"claim_id": "c1", "raw_claim_text": "bati o carro"}, context=context
    )

    assert out["context_sufficient"] is False
    assert out["citations"] == []
    assert [e.node for e in out["audit_trail"]] == ["intake", "retrieval"]


@pytest.mark.unit
def test_incomplete_claim_loops_then_terminates_as_exhausted() -> None:
    out = _invoke(missing=["data_evento_vigencia"])

    assert out["clarification_exhausted"] is True
    assert out["clarification_rounds"] == MAX_CLARIFICATION_ROUNDS
    assert out["missing_information"] == ["data_evento_vigencia"]
    assert len(out["clarification_questions"]) == MAX_CLARIFICATION_ROUNDS
    assert out["audit_trail"][-1].node == "clarification_exhausted"
    # intake (x3) + clarification (x2) + clarification_exhausted (x1)
    node_runs = [e.node for e in out["audit_trail"]]
    assert node_runs.count("intake") == MAX_CLARIFICATION_ROUNDS + 1
    assert node_runs.count("clarification") == MAX_CLARIFICATION_ROUNDS


def _incomplete_claims() -> list[SyntheticClaim]:
    rows = [
        SyntheticClaim.model_validate_json(line)
        for line in CLAIMS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return [r for r in rows if r.missing_fact_type is not None]


@pytest.mark.unit
def test_every_incomplete_m2_04_claim_terminates() -> None:
    claims = _incomplete_claims()
    assert len(claims) >= 10, "DoD: at least 10 incomplete claims from [M2-04]"

    for claim in claims:
        tag = cast(Any, claim.missing_fact_type).value
        compiled = build_claim_graph().compile()
        out = compiled.invoke(
            {"claim_id": claim.claim_id, "raw_claim_text": claim.narrative},
            context=_context([tag]),
        )
        assert out["clarification_exhausted"] is True, claim.claim_id
        assert out["clarification_rounds"] == MAX_CLARIFICATION_ROUNDS, claim.claim_id
