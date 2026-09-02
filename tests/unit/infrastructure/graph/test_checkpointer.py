"""Unit tests for the checkpointer wiring ([M4-09]).

No database: the URL conversion, the readiness probe (over a fake connection)
and -- the one that earns its place -- the serializer allowlist.

That last one guards a silent failure. LangGraph's msgpack serializer will not
rebuild a type it was not told about, and it does not raise when it meets one:
the value comes back as a plain ``dict``. A ``ClaimState`` restored from a
checkpoint would then hand nodes mappings where they expect models, several
supersteps away from the cause. So a fully populated state is round-tripped here
and every value is checked for its type, and the allowlist is derived from
``state.py`` rather than hand-listed so a sub-model added later is covered
without anyone remembering to come back.
"""

import ast
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from domain.clause_classification import ClauseType
from domain.verdict import Verdict
from infrastructure.graph.checkpointer import (
    CHECKPOINT_TABLE,
    CHECKPOINTER_TABLES,
    SETUP_COMMAND,
    assert_checkpointer_ready,
    build_checkpoint_serializer,
    checkpoint_allowed_types,
    checkpointer_conn_string,
)
from infrastructure.graph.state import (
    AuditEvent,
    Citation,
    ClaimState,
    ClarificationQuestion,
    CompatibilityAssessment,
    ConsistencyReport,
    ConsistencySignal,
    ExtractedEntities,
    HumanDecision,
    Recommendation,
    TokenUsage,
)

# --- the connection string ------------------------------------------------


@pytest.mark.unit
def test_the_sqlalchemy_driver_suffix_is_dropped() -> None:
    # psycopg does not accept `postgresql+psycopg://`, and the project's
    # settings produce nothing else.
    assert (
        checkpointer_conn_string("postgresql+psycopg://u:p@localhost:5432/db")
        == "postgresql://u:p@localhost:5432/db"
    )


@pytest.mark.unit
def test_a_plain_postgres_url_passes_through() -> None:
    assert (
        checkpointer_conn_string("postgresql://u:p@localhost:5432/db")
        == "postgresql://u:p@localhost:5432/db"
    )


@pytest.mark.unit
def test_the_password_survives_the_conversion() -> None:
    # Rendered in the clear on purpose -- it is going straight into a connection
    # call, and SQLAlchemy hides it by default.
    converted = checkpointer_conn_string("postgresql+psycopg://u:s3cr%40t@h:5432/db")
    assert "s3cr%40t" in converted


# --- the tables Alembic must not touch -------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[4]
ENV_PY = REPO_ROOT / "alembic" / "env.py"


def _env_unmanaged_tables() -> set[str]:
    """The literal ``UNMANAGED_TABLES`` set from ``alembic/env.py``.

    Read with ``ast`` rather than imported: ``env.py`` runs migrations at import
    time.
    """
    tree = ast.parse(ENV_PY.read_text(encoding="utf-8"), filename=str(ENV_PY))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "UNMANAGED_TABLES" for t in node.targets
        ):
            call = node.value
            assert isinstance(call, ast.Call)
            return set(ast.literal_eval(call.args[0]))
    raise AssertionError("no UNMANAGED_TABLES assignment in alembic/env.py")


@pytest.mark.unit
def test_alembic_is_told_to_ignore_every_checkpointer_table() -> None:
    # Without the filter, `alembic revision --autogenerate` sees these in the
    # database but not in Base.metadata and writes a migration that DROPS the
    # checkpointer -- and every paused run with it. The literal in env.py is
    # kept there so that file imports no app code; this ties the two together.
    assert _env_unmanaged_tables() == set(CHECKPOINTER_TABLES)


@pytest.mark.unit
def test_the_probed_table_is_one_the_checkpointer_creates() -> None:
    assert CHECKPOINT_TABLE in CHECKPOINTER_TABLES


# --- the readiness probe ---------------------------------------------------


class _FakeCursor:
    def __init__(self, row: object | None) -> None:
        self._row = row
        self.queries: list[tuple[str, object]] = []

    def __enter__(self) -> "_FakeCursor":
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def execute(self, query: str, params: object = None) -> None:
        self.queries.append((query, params))

    def fetchone(self) -> object | None:
        return self._row


class _FakeConnection:
    def __init__(self, row: object | None) -> None:
        self.cursor_obj = _FakeCursor(row)

    def cursor(self) -> _FakeCursor:
        return self.cursor_obj


@pytest.mark.unit
def test_a_missing_checkpoint_table_names_the_fix() -> None:
    conn = _FakeConnection(None)
    with pytest.raises(RuntimeError) as excinfo:
        assert_checkpointer_ready(cast(Any, conn))
    message = str(excinfo.value)
    assert CHECKPOINT_TABLE in message
    assert SETUP_COMMAND in message


@pytest.mark.unit
def test_a_present_checkpoint_table_passes() -> None:
    conn = _FakeConnection({"?column?": 1})
    assert_checkpointer_ready(cast(Any, conn))
    assert conn.cursor_obj.queries[0][1] == (CHECKPOINT_TABLE,)


# --- the serializer allowlist ---------------------------------------------


def _populated_state() -> ClaimState:
    """A ``ClaimState`` with every optional channel filled, models and all."""
    citation = Citation(
        clause_id="doc-1:1.1",
        document_id="doc-1",
        susep_process="15414.900000/2013-00",
        clause_type=ClauseType.EXCLUSION,
        relevance_score=0.87,
        excerpt="A seguradora não cobre corrida em pista.",
    )
    recommendation = Recommendation(
        recommended_action="Encaminhar para revisão humana.",
        justification="Resumo.",
        citations=[citation],
        consistency_flags=[
            ConsistencySignal(
                check="amount_implausibly_high",
                severity="attention",
                detail="valor acima da banda",
                source="deterministic",
            )
        ],
        confidence=0.55,
    )
    return {
        "claim_id": "c1",
        "raw_claim_text": "bati o carro",
        "entities": ExtractedEntities(
            event_type="colisão",
            event_date="2026-01-05",
            estimated_amount=12500.0,
            product_line="CASCO",
        ),
        "missing_information": ["data_evento_vigencia"],
        "clarification_rounds": 1,
        "clarification_questions": [
            ClarificationQuestion(field="data_evento_vigencia", question="Quando?")
        ],
        "clarification_exhausted": False,
        "citations": [citation],
        "context_sufficient": True,
        "compatibility": CompatibilityAssessment(
            verdict=Verdict.INCOMPATIBLE,
            reasoning="A cláusula 1.1 exclui o evento.",
            citations=[citation],
            confidence=0.72,
        ),
        "consistency": ConsistencyReport(signals=[]),
        "recommendation": recommendation,
        "human_decision": HumanDecision(
            decision="edit",
            notes="ajustei a justificativa",
            decided_at=datetime(2026, 9, 2, 12, 0, tzinfo=UTC),
            edited_recommendation=recommendation,
        ),
        "audit_trail": [
            AuditEvent(
                node="compatibility",
                action="assess",
                model="fake-reasoning-model",
                token_usage=TokenUsage(
                    input_tokens=100, output_tokens=20, total_tokens=120
                ),
                confidence=0.72,
            )
        ],
    }


@pytest.mark.unit
def test_a_full_claim_state_survives_the_checkpoint_round_trip() -> None:
    serde = build_checkpoint_serializer()
    state = _populated_state()

    restored = cast(ClaimState, serde.loads_typed(serde.dumps_typed(state)))

    assert restored == state
    # Equality alone would not catch the failure this guards: a blocked type
    # comes back as a `dict`, and a dict never equals a model -- but a channel
    # holding a *list* of them is easy to eyeball wrong. Check the types.
    assert isinstance(restored["recommendation"], Recommendation)
    assert isinstance(restored["human_decision"], HumanDecision)
    assert isinstance(restored["compatibility"], CompatibilityAssessment)
    assert isinstance(restored["consistency"], ConsistencyReport)
    assert isinstance(restored["entities"], ExtractedEntities)
    assert isinstance(restored["citations"][0], Citation)
    assert isinstance(restored["clarification_questions"][0], ClarificationQuestion)
    assert isinstance(restored["audit_trail"][0], AuditEvent)
    assert isinstance(restored["audit_trail"][0].token_usage, TokenUsage)
    # the enums, which live in `domain` and are allow-listed by hand
    assert restored["citations"][0].clause_type is ClauseType.EXCLUSION
    assert restored["compatibility"].verdict is Verdict.INCOMPATIBLE


@pytest.mark.unit
def test_the_allowlist_is_derived_from_the_state_module() -> None:
    allowed = checkpoint_allowed_types()
    names = {name for module, name in allowed if module.endswith("graph.state")}
    # Every model `state.py` defines, found by enumeration rather than listed.
    assert {
        "AuditEvent",
        "Citation",
        "ClarificationQuestion",
        "CompatibilityAssessment",
        "ConsistencyReport",
        "ConsistencySignal",
        "ExtractedEntities",
        "HumanDecision",
        "Recommendation",
        "TokenUsage",
    } <= names
    assert ("domain.verdict", "Verdict") in allowed
    assert ("domain.clause_classification", "ClauseType") in allowed


@pytest.mark.unit
def test_an_incomplete_allowlist_degrades_silently() -> None:
    # The behaviour the test above exists to catch: no exception, just a dict
    # where a model should be. Pinned here so the guard's premise is on record.
    partial = JsonPlusSerializer(
        allowed_msgpack_modules=[("infrastructure.graph.state", "Citation")]
    )
    restored = partial.loads_typed(partial.dumps_typed(_populated_state()))

    assert isinstance(restored["recommendation"], dict)
