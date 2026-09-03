"""The compiled claim graph: the loop and the retrieval hand-off ([M4-03]/[M4-04]).

``route_after_intake`` / ``route_after_retrieval`` in isolation, the graph's
structure, and -- the [M4-03] DoD's core check -- that the
``intake ⇄ clarification`` loop terminates for every incomplete claim in the
M2-04 set. Termination is a property of the router plus
``MAX_CLARIFICATION_ROUNDS``, so a fake LLM plus a stub retriever is the right
tool: no network, and no reliance on LangGraph's ``recursion_limit``.

Since [M4-09] the graph ends on the human checkpoint, so every run here compiles
with an ``InMemorySaver`` and a ``thread_id`` and resumes with a decision. That
is not ceremony: without a checkpointer LangGraph returns silently at the
``interrupt()`` instead of failing, and every assertion below about what the
graph produced would be testing a truncated run.
"""

from pathlib import Path
from typing import Any, cast

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable, RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START
from langgraph.types import Command

from domain.clause_classification import ClauseType
from domain.verdict import Verdict
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
    CompatibilityOutput,
    ConsistencyOutput,
    IntakeOutput,
    ReasonedAssertion,
    RecommendationOutput,
)
from infrastructure.graph.state import (
    ConsistencyReport,
    HumanDecision,
    Recommendation,
)
from infrastructure.rag.retrieved_clause import RetrievedClause

REPO_ROOT = Path(__file__).resolve().parents[4]
CLAIMS_PATH = REPO_ROOT / "data" / "synthetic_claims" / "claims.jsonl"


class _FakeRaw:
    usage_metadata = None


class _LoopModel:
    """Serves every node: intake always reports ``missing`` as unresolved."""

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
            elif schema is CompatibilityOutput:
                parsed = CompatibilityOutput(
                    verdict="compatible",
                    assertions=[
                        ReasonedAssertion(
                            statement="A colisão está coberta.",
                            clause_ids=[_STUB_HIT.clause_id],
                        )
                    ],
                    confidence=0.7,
                )
            elif schema is ConsistencyOutput:
                parsed = ConsistencyOutput(signals=[])
            elif schema is RecommendationOutput:
                parsed = RecommendationOutput(justification="Resumo para o revisor.")
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


APPROVE: dict[str, Any] = {"decision": "approve", "notes": "conferido"}


def _run(
    context: GraphContext,
    *,
    claim_id: str = "c1",
    text: str = "bati o carro",
    resume: dict[str, Any] | None = APPROVE,
) -> dict[str, Any]:
    """Run the graph to the [M4-09] checkpoint and, unless told not to, past it."""
    compiled = build_claim_graph().compile(checkpointer=InMemorySaver())
    config: Any = {"configurable": {"thread_id": f"thread-{claim_id}"}}
    out = compiled.invoke(
        {"claim_id": claim_id, "raw_claim_text": text}, config=config, context=context
    )
    assert "__interrupt__" in out, "every path must reach the human checkpoint"
    if resume is None:
        return out
    return compiled.invoke(Command(resume=resume), config=config, context=context)


def _invoke(missing: list[str]) -> dict[str, Any]:
    return _run(_context(missing))


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
def test_route_after_retrieval_fans_out_to_both_nodes_when_context_suffices() -> None:
    state = {"claim_id": "c", "raw_claim_text": "x", "context_sufficient": True}
    assert route_after_retrieval(cast(Any, state)) == ["compatibility", "consistency"]


@pytest.mark.unit
def test_route_after_retrieval_sends_insufficient_context_to_recommendation() -> None:
    state = {"claim_id": "c", "raw_claim_text": "x", "context_sufficient": False}
    assert route_after_retrieval(cast(Any, state)) == ["recommendation"]


# --- structure --------------------------------------------------------


@pytest.mark.unit
def test_graph_has_the_expected_nodes_and_entry() -> None:
    graph = build_claim_graph().compile().get_graph()
    assert {
        "intake",
        "clarification",
        "clarification_exhausted",
        "retrieval",
        "compatibility",
        "consistency",
        "recommendation",
        "human_review",
    } <= set(graph.nodes)
    assert any(e.source == START and e.target == "intake" for e in graph.edges)
    assert any(e.source == "intake" and e.target == "retrieval" for e in graph.edges)


@pytest.mark.unit
def test_every_terminal_path_converges_on_the_recommendation_node() -> None:
    # [M4-07] fanned retrieval out to the two assessment nodes; [M4-08] repointed
    # every terminal edge at the recommendation node; [M4-09] put the human
    # checkpoint behind it, so that node alone reaches END.
    graph = build_claim_graph().compile().get_graph()
    edges = {(e.source, e.target) for e in graph.edges}
    assert ("retrieval", "compatibility") in edges
    assert ("retrieval", "consistency") in edges
    assert ("retrieval", "recommendation") in edges
    assert ("compatibility", "recommendation") in edges
    assert ("consistency", "recommendation") in edges
    assert ("clarification_exhausted", "recommendation") in edges
    assert ("recommendation", "human_review") in edges
    assert ("human_review", END) in edges
    assert not any(src != "human_review" and tgt == END for src, tgt in edges), (
        "nothing may reach END without passing the human checkpoint"
    )


# --- the loop --------------------------------------------------------


@pytest.mark.unit
def test_complete_claim_passes_straight_through() -> None:
    out = _invoke(missing=[])

    assert out.get("clarification_exhausted") in (None, False)
    assert not out.get("clarification_questions")
    assert out.get("clarification_rounds", 0) == 0
    # intake -> retrieval -> then the parallel superstep: compatibility (one
    # event) and consistency (two, one per leg). Order within the superstep
    # follows the route list, but assert on counts to stay robust.
    node_runs = [e.node for e in out["audit_trail"]]
    assert node_runs[:2] == ["intake", "retrieval"]
    assert node_runs.count("compatibility") == 1
    assert node_runs.count("consistency") == 2
    # the recommendation node runs once, after both assessment branches, and
    # the human checkpoint closes the run
    assert node_runs.count("recommendation") == 1
    assert node_runs[-2:] == ["recommendation", "human_review"]
    assert out["context_sufficient"] is True
    assert [c.clause_id for c in out["citations"]] == ["doc-1:1.1"]
    assert out["compatibility"].verdict is Verdict.COMPATIBLE
    assert [c.clause_id for c in out["compatibility"].citations] == ["doc-1:1.1"]
    assert isinstance(out["consistency"], ConsistencyReport)
    assert out["consistency"].signals == []
    rec = out["recommendation"]
    assert isinstance(rec, Recommendation)
    assert [c.clause_id for c in rec.citations] == ["doc-1:1.1"]
    assert rec.consistency_flags == []
    assert rec.justification == "Resumo para o revisor."
    decision = out["human_decision"]
    assert isinstance(decision, HumanDecision)
    assert decision.decision == "approve"


@pytest.mark.unit
def test_retrieval_with_no_hits_flags_insufficient_context() -> None:
    class _EmptyRetriever:
        def retrieve(
            self, question: str, *, k: int, metadata_filter: object | None = None
        ) -> list[RetrievedClause]:
            return []

    base = _context(missing=[])
    context = GraphContext(
        fast_model=base.fast_model,
        reasoning_model=base.reasoning_model,
        retriever=cast(RetrievalPort, _EmptyRetriever()),
        llm_settings=base.llm_settings,
    )
    out = _run(context)

    assert out["context_sufficient"] is False
    assert out["citations"] == []
    assert [e.node for e in out["audit_trail"]] == [
        "intake",
        "retrieval",
        "recommendation",
        "human_review",
    ]
    rec = out["recommendation"]
    assert isinstance(rec, Recommendation)
    assert rec.citations == []
    assert "revisão manual de cláusulas" in rec.recommended_action


@pytest.mark.unit
def test_a_failure_in_one_assessment_branch_is_not_silently_swallowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # [M4-07] DoD: a failure in one parallel branch must not silently truncate
    # the other. With no error_handler the superstep aborts loudly -- invoke()
    # re-raises and no partial state dict (consistency set, compatibility
    # missing) is ever returned.
    monkeypatch.setattr("time.sleep", lambda *_a, **_k: None)

    class _BoomChain:
        def invoke(self, _messages: Any) -> dict[str, object]:
            raise RuntimeError("reasoning model is down")

    class _BoomReasoningModel:
        def with_structured_output(
            self, schema: type, include_raw: bool = False
        ) -> Any:
            return _BoomChain()

    base = _context(missing=[])
    context = GraphContext(
        fast_model=base.fast_model,
        reasoning_model=cast(BaseChatModel, _BoomReasoningModel()),
        retriever=base.retriever,
        llm_settings=base.llm_settings,
    )
    with pytest.raises(RuntimeError, match="reasoning model is down"):
        _run(context)


@pytest.mark.unit
def test_incomplete_claim_loops_then_terminates_as_exhausted() -> None:
    out = _invoke(missing=["data_evento_vigencia"])

    assert out["clarification_exhausted"] is True
    assert out["clarification_rounds"] == MAX_CLARIFICATION_ROUNDS
    assert out["missing_information"] == ["data_evento_vigencia"]
    assert len(out["clarification_questions"]) == MAX_CLARIFICATION_ROUNDS
    # clarification_exhausted now hands off to the recommendation node
    assert out["audit_trail"][-3].node == "clarification_exhausted"
    assert out["audit_trail"][-2].node == "recommendation"
    assert out["audit_trail"][-1].node == "human_review"
    rec = out["recommendation"]
    assert isinstance(rec, Recommendation)
    assert "data_evento_vigencia" in rec.recommended_action
    # intake x3 + clarification x2 + clarification_exhausted x1 + recommendation x1
    node_runs = [e.node for e in out["audit_trail"]]
    assert node_runs.count("intake") == MAX_CLARIFICATION_ROUNDS + 1
    assert node_runs.count("clarification") == MAX_CLARIFICATION_ROUNDS
    assert node_runs.count("recommendation") == 1


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
        out = _run(_context([tag]), claim_id=claim.claim_id, text=claim.narrative)
        assert out["clarification_exhausted"] is True, claim.claim_id
        assert out["clarification_rounds"] == MAX_CLARIFICATION_ROUNDS, claim.claim_id
        assert isinstance(out["recommendation"], Recommendation), claim.claim_id
        assert isinstance(out["human_decision"], HumanDecision), claim.claim_id
