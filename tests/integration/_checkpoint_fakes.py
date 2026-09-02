"""The fake graph context the [M4-09] restart test and its worker both build.

Shared so the two processes are demonstrably running the *same* graph -- the
point of the restart test is that only the database is carried between them, and
that only holds if neither side quietly builds something different.

No network: a fake chat model serving one canned structured output per node, and
a stub retriever returning one high-scoring clause so the [M3-07] gate is
satisfied and the run reaches the checkpoint through the full assessment path.
"""

from typing import Any, cast

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable, RunnableLambda
from sqlalchemy.orm import Session, sessionmaker

from domain.clause_classification import ClauseType
from infrastructure.config.enums import LlmProvider
from infrastructure.config.settings import LlmSettings
from infrastructure.database.graph_audit_sink import SqlAlchemyAuditTrailSink
from infrastructure.graph.context import GraphContext, RetrievalPort
from infrastructure.graph.schemas import (
    ClarificationOutput,
    CompatibilityOutput,
    ConsistencyOutput,
    IntakeOutput,
    ReasonedAssertion,
    RecommendationOutput,
)
from infrastructure.rag.retrieved_clause import RetrievedClause

CLAUSE_ID = "doc-1:1.1"
JUSTIFICATION = "Resumo para o revisor."

STUB_HIT = RetrievedClause(
    clause_id=CLAUSE_ID,
    document_id="doc-1",
    susep_process="15414.900000/2013-00",
    clause_type=ClauseType.COVERAGE,
    excerpt="A seguradora cobre colisao.",
    score=0.95,
)


class _FakeRaw:
    usage_metadata = None


class _FakeModel:
    """Serves every LLM node in the graph with a canned structured output."""

    def with_structured_output(
        self, schema: type, include_raw: bool = False
    ) -> Runnable[Any, Any]:
        def _invoke(_messages: Any) -> dict[str, object]:
            if schema is IntakeOutput:
                parsed: object = IntakeOutput(
                    product_line="CASCO", missing_information=[]
                )
            elif schema is CompatibilityOutput:
                parsed = CompatibilityOutput(
                    verdict="compatible",
                    assertions=[
                        ReasonedAssertion(
                            statement="A colisão está coberta.",
                            clause_ids=[CLAUSE_ID],
                        )
                    ],
                    confidence=0.7,
                )
            elif schema is ConsistencyOutput:
                parsed = ConsistencyOutput(signals=[])
            elif schema is RecommendationOutput:
                parsed = RecommendationOutput(justification=JUSTIFICATION)
            else:
                parsed = ClarificationOutput(questions=[])
            return {"parsed": parsed, "raw": _FakeRaw()}

        return RunnableLambda(_invoke)


class _StubRetriever:
    """One high-scoring clause, so the [M3-07] gate reports sufficient context."""

    def retrieve(
        self, question: str, *, k: int, metadata_filter: object | None = None
    ) -> list[RetrievedClause]:
        return [STUB_HIT]


def build_fake_context(
    session_factory: sessionmaker[Session] | None = None,
) -> GraphContext:
    """A ``GraphContext`` with fakes for everything but the audit sink.

    ``session_factory`` is the real one when the caller wants the durable audit
    write exercised; ``None`` leaves ``audit_sink`` unset.
    """
    model = cast(BaseChatModel, _FakeModel())
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
        audit_sink=(
            SqlAlchemyAuditTrailSink(session_factory)
            if session_factory is not None
            else None
        ),
    )
