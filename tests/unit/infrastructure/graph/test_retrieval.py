"""Unit tests for the retrieval node ([M4-04]).

A fake ``RetrievalPort`` -- no DB, no models, no compiled graph except the one
test that proves the ``Runtime[GraphContext]`` injection path. Retrieval quality
against the golden set is a separate ``eval``-marked measurement
(``scripts/eval_retrieval_node.py`` / ``tests/eval``).
"""

from typing import Any, cast

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from domain.clause_classification import ClauseType
from infrastructure.config.enums import LlmProvider
from infrastructure.config.settings import LlmSettings
from infrastructure.graph.context import GraphContext, RetrievalPort
from infrastructure.graph.nodes.retrieval import (
    RETRIEVAL_K,
    _build_filter,
    _build_query,
    retrieval,
)
from infrastructure.graph.state import Citation, ClaimState, ExtractedEntities
from infrastructure.rag.retrieval_filter import RetrievalFilter
from infrastructure.rag.retrieved_clause import RetrievedClause


class _RecordingRetriever:
    """A ``RetrievalPort`` that records its call args and returns canned hits."""

    def __init__(self, hits: list[RetrievedClause]) -> None:
        self._hits = hits
        self.question: str | None = None
        self.k: int | None = None
        self.metadata_filter: RetrievalFilter | None = None

    def retrieve(
        self,
        question: str,
        *,
        k: int,
        metadata_filter: RetrievalFilter | None = None,
    ) -> list[RetrievedClause]:
        self.question = question
        self.k = k
        self.metadata_filter = metadata_filter
        return list(self._hits)


def _hit(
    clause_id: str, score: float, *, clause_type: ClauseType = ClauseType.COVERAGE
) -> RetrievedClause:
    return RetrievedClause(
        clause_id=clause_id,
        document_id="doc-1",
        susep_process="15414.900000/2013-00",
        clause_type=clause_type,
        excerpt=f"Texto da {clause_id}.",
        score=score,
    )


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


def _context(retriever: RetrievalPort) -> GraphContext:
    unused = cast(BaseChatModel, object())
    return GraphContext(
        fast_model=unused,
        reasoning_model=unused,
        retriever=retriever,
        llm_settings=_llm_settings(),
    )


def _run(
    retriever: _RecordingRetriever,
    *,
    entities: ExtractedEntities | None,
    raw_claim_text: str = "bati o carro no portao ontem",
) -> dict[str, object]:
    state: dict[str, object] = {"claim_id": "c1", "raw_claim_text": raw_claim_text}
    if entities is not None:
        state["entities"] = entities
    return retrieval(
        cast(ClaimState, state),
        Runtime(context=_context(cast(RetrievalPort, retriever))),
    )


# --- query building -------------------------------------------------


@pytest.mark.unit
def test_build_query_joins_entity_fields_not_the_raw_text() -> None:
    entities = ExtractedEntities(
        event_type="colisão",
        description="bateu contra uma mureta",
        vehicle_info="carro de passeio",
        estimated_amount=15000.0,
    )
    query = _build_query(entities, fallback="RAW TEXT")
    assert query == "colisão bateu contra uma mureta carro de passeio"
    assert "RAW TEXT" not in query


@pytest.mark.unit
def test_build_query_falls_back_only_when_entities_carry_no_text() -> None:
    assert _build_query(None, fallback="raw") == "raw"
    assert _build_query(ExtractedEntities(), fallback="raw") == "raw"
    assert (
        _build_query(ExtractedEntities(estimated_amount=1.0), fallback="raw") == "raw"
    )


# --- filter building -----------------------------------------------


@pytest.mark.unit
def test_build_filter_uses_the_classification() -> None:
    entities = ExtractedEntities(
        susep_process="15414.900000/2013-00", product_line="CASCO"
    )
    assert _build_filter(entities) == RetrievalFilter(
        susep_process="15414.900000/2013-00", product_line="CASCO"
    )


@pytest.mark.unit
def test_build_filter_is_none_when_nothing_is_known() -> None:
    assert _build_filter(None) is None
    assert _build_filter(ExtractedEntities(event_type="colisão")) is None


# --- the node ----------------------------------------------------


@pytest.mark.unit
def test_retrieval_passes_the_built_query_and_filter_to_the_port() -> None:
    retriever = _RecordingRetriever([_hit("doc-1:1.1", 0.9)])
    _run(
        retriever,
        entities=ExtractedEntities(
            description="colisão com mureta",
            susep_process="15414.900000/2013-00",
            product_line="CASCO",
        ),
    )
    assert retriever.question == "colisão com mureta"
    assert retriever.k == RETRIEVAL_K
    assert retriever.metadata_filter == RetrievalFilter(
        susep_process="15414.900000/2013-00", product_line="CASCO"
    )


@pytest.mark.unit
def test_retrieval_hydrates_citations_from_the_port_hits() -> None:
    retriever = _RecordingRetriever(
        [
            _hit("doc-1:1.1", 0.91),
            _hit("doc-1:2.4", 0.55, clause_type=ClauseType.EXCLUSION),
        ]
    )
    out = _run(retriever, entities=ExtractedEntities(description="colisão"))

    citations = cast(list[Citation], out["citations"])
    assert [c.clause_id for c in citations] == ["doc-1:1.1", "doc-1:2.4"]
    assert citations[0].relevance_score == pytest.approx(0.91)
    assert citations[0].excerpt == "Texto da doc-1:1.1."
    assert citations[0].document_id == "doc-1"
    assert citations[1].clause_type is ClauseType.EXCLUSION


@pytest.mark.unit
def test_retrieval_flags_context_sufficient_when_the_top_score_is_high() -> None:
    retriever = _RecordingRetriever([_hit("doc-1:1.1", 0.92)])
    out = _run(retriever, entities=ExtractedEntities(description="colisão"))
    assert out["context_sufficient"] is True


@pytest.mark.unit
def test_retrieval_abstains_when_nothing_is_retrieved() -> None:
    retriever = _RecordingRetriever([])
    out = _run(retriever, entities=ExtractedEntities(description="colisão"))
    assert out["context_sufficient"] is False
    assert out["citations"] == []


@pytest.mark.unit
def test_retrieval_abstains_when_the_top_score_is_below_the_floor() -> None:
    retriever = _RecordingRetriever([_hit("doc-1:1.1", 0.20)])
    out = _run(retriever, entities=ExtractedEntities(description="colisão"))
    assert out["context_sufficient"] is False


@pytest.mark.unit
def test_retrieval_records_a_model_free_audit_event() -> None:
    retriever = _RecordingRetriever([_hit("doc-1:1.1", 0.92)])
    out = _run(retriever, entities=ExtractedEntities(description="colisão"))

    trail = cast(list[Any], out["audit_trail"])
    assert len(trail) == 1
    assert trail[0].node == "retrieval"
    assert trail[0].action == "retrieve_clauses"
    assert trail[0].model is None
    assert trail[0].token_usage is None
    assert trail[0].confidence is None


@pytest.mark.unit
def test_retrieval_runs_as_a_node_in_a_compiled_state_graph() -> None:
    retriever = _RecordingRetriever([_hit("doc-1:1.1", 0.92)])
    builder: Any = StateGraph(ClaimState, context_schema=GraphContext)
    builder.add_node("retrieval", retrieval)
    builder.add_edge(START, "retrieval")
    builder.add_edge("retrieval", END)
    compiled = builder.compile()

    out = compiled.invoke(
        {
            "claim_id": "c1",
            "raw_claim_text": "bati o carro",
            "entities": ExtractedEntities(description="colisão com mureta"),
        },
        context=_context(cast(RetrievalPort, retriever)),
    )
    assert [c.clause_id for c in out["citations"]] == ["doc-1:1.1"]
    assert out["context_sufficient"] is True
