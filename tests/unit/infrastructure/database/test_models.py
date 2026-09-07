"""Schema-level guarantees for the ORM models.

``chunk`` -- [M3-02]; ``audit_event`` -- [M4-09]; ``assessment`` /
``human_decision`` -- [M5-03]; ``assessment_job`` -- [M5-05].
"""

import re

import pytest
from pgvector.sqlalchemy import HALFVEC
from sqlalchemy import CheckConstraint, DateTime, Table

from application.assessment_job import JobStatus
from application.assessment_record import AssessmentStatus
from domain.chunk import ChunkRule
from domain.clause_classification import ClauseType, TypeSource
from domain.human_decision import DecisionOutcome
from domain.verdict import Verdict
from infrastructure.database.models import (
    AssessmentJobRow,
    AssessmentRow,
    AuditEventRow,
    ChunkRow,
    HumanDecisionRow,
)
from infrastructure.graph.state import AuditEvent
from infrastructure.rag.embedding_config import EMBEDDING_DIMENSIONS

# A properly-typed Table handle (``ChunkRow.__table__`` is typed as a bare
# FromClause).
_CHUNK_TABLE = ChunkRow.metadata.tables["chunk"]
_AUDIT_TABLE = AuditEventRow.metadata.tables["audit_event"]
_ASSESSMENT_TABLE = AssessmentRow.metadata.tables["assessment"]
_DECISION_TABLE = HumanDecisionRow.metadata.tables["human_decision"]
_JOB_TABLE = AssessmentJobRow.metadata.tables["assessment_job"]

_EXPECTED_INDEXES = {
    "ix_chunk_clause_type",
    "ix_chunk_bundle_section",
    "ix_chunk_susep_process",
    "ix_chunk_cnpj",
    "ix_chunk_product_line",
    "ix_chunk_susep_process_cnpj",
}


def _check_constraint_values(name: str, table: Table = _CHUNK_TABLE) -> set[str]:
    for constraint in table.constraints:
        if isinstance(constraint, CheckConstraint) and str(constraint.name) == name:
            return set(re.findall(r"'([^']*)'", str(constraint.sqltext)))
    raise AssertionError(f"no CHECK constraint named {name}")


@pytest.mark.unit
def test_table_name_and_primary_key() -> None:
    assert ChunkRow.__tablename__ == "chunk"
    assert [column.name for column in _CHUNK_TABLE.primary_key.columns] == ["chunk_id"]


@pytest.mark.unit
def test_bundle_section_is_genuinely_nullable_with_no_sentinel() -> None:
    column = _CHUNK_TABLE.c.bundle_section

    assert column.nullable is True
    # A server_default or a Python default would be a sentinel by another name;
    # M3-04's strict `WHERE bundle_section = :x` must be able to exclude these
    # rows in plain SQL.
    assert column.server_default is None
    assert column.default is None


@pytest.mark.unit
def test_confidence_is_nullable() -> None:
    assert _CHUNK_TABLE.c.confidence.nullable is True


@pytest.mark.unit
def test_provenance_and_attribution_columns_are_present_and_not_null() -> None:
    required = [
        "document_id",
        "clause_id",
        "susep_process",
        "insurer",
        "cnpj",
        "product_line",
        "indemnity_regime",
        "filing_year",
        "source",
        "type_source",
    ]
    for name in required:
        assert _CHUNK_TABLE.c[name].nullable is False


@pytest.mark.unit
def test_embedded_and_display_text_are_separate_columns() -> None:
    assert "embedded_text" in _CHUNK_TABLE.c
    assert "display_text" in _CHUNK_TABLE.c


@pytest.mark.unit
def test_embedding_is_a_nullable_halfvec_of_the_pinned_dimension() -> None:
    column = _CHUNK_TABLE.c.embedding

    # Width comes from the pinned model contract, not a literal here -- the
    # migration's literal `768` is caught by `alembic check` if it drifts.
    assert isinstance(column.type, HALFVEC)
    assert column.type.dim == EMBEDDING_DIMENSIONS
    # An un-embedded chunk is `NULL`; that is the pipeline's resumable cursor,
    # so no server_default / Python default may fill it in.
    assert column.nullable is True
    assert column.server_default is None
    assert column.default is None


@pytest.mark.unit
def test_embedding_column_is_not_indexed_on_the_model() -> None:
    # The HNSW ANN index over `embedding` is created imperatively by
    # `infrastructure.rag.ann_index`, never declared on the model -- M3-02's
    # benchmark chose exact `<=>` over the metadata-filtered partition as the
    # default path (docs/EMBEDDINGS.md).
    indexed_columns = {
        column.name for index in _CHUNK_TABLE.indexes for column in index.columns
    }
    assert "embedding" not in indexed_columns


@pytest.mark.unit
def test_check_constraints_match_the_domain_enums() -> None:
    assert _check_constraint_values("ck_chunk_rule_valid") == {
        member.value for member in ChunkRule
    }
    assert _check_constraint_values("ck_chunk_clause_type_valid") == {
        member.value for member in ClauseType
    }
    assert _check_constraint_values("ck_chunk_type_source_valid") == {
        member.value for member in TypeSource
    }
    assert _check_constraint_values("ck_chunk_source_valid") == {"text", "ocr"}


@pytest.mark.unit
def test_filter_indexes_are_declared() -> None:
    declared = {str(index.name) for index in _CHUNK_TABLE.indexes}
    assert declared == _EXPECTED_INDEXES


@pytest.mark.unit
def test_insurer_is_not_indexed() -> None:
    # M3-04 filters insurers by CNPJ, never by name.
    indexed_columns = {
        column.name for index in _CHUNK_TABLE.indexes for column in index.columns
    }
    assert "insurer" not in indexed_columns


# --- audit_event -- [M4-09] ------------------------------------------------


@pytest.mark.unit
def test_audit_event_is_keyed_by_thread_and_position() -> None:
    # The composite key is the whole idempotency story: the checkpoint node
    # re-runs on every resume, and a record's index in its thread's trail is
    # deterministic, so repeating the write is a no-op rather than a duplicate.
    assert AuditEventRow.__tablename__ == "audit_event"
    assert [column.name for column in _AUDIT_TABLE.primary_key.columns] == [
        "thread_id",
        "sequence",
    ]


@pytest.mark.unit
def test_audit_event_mirrors_every_audit_event_field() -> None:
    columns = set(_AUDIT_TABLE.c.keys())
    flattened = {"input_tokens", "output_tokens", "total_tokens"}
    # `token_usage` is the one field that does not map one-to-one: a closed
    # three-field record, flattened so the trail stays queryable with plain
    # aggregates.
    expected = (set(AuditEvent.model_fields) - {"token_usage"}) | flattened
    assert expected <= columns


@pytest.mark.unit
def test_audit_event_carries_the_run_identity_and_the_decision_payload() -> None:
    assert {"thread_id", "sequence", "claim_id", "payload"} <= set(
        _AUDIT_TABLE.c.keys()
    )
    # Optional: only the human-review event fills it.
    assert _AUDIT_TABLE.c.payload.nullable is True
    assert _AUDIT_TABLE.c.claim_id.nullable is False


@pytest.mark.unit
def test_audit_event_indexes_the_column_a_human_searches_by() -> None:
    # `thread_id` is already covered by the primary key.
    assert {str(index.name) for index in _AUDIT_TABLE.indexes} == {
        "ix_audit_event_claim_id"
    }


@pytest.mark.unit
def test_audit_event_timestamp_is_timezone_aware() -> None:
    # `AuditEvent.timestamp` defaults to `datetime.now(UTC)`; a naive column
    # would silently drop the offset.
    column_type = _AUDIT_TABLE.c.timestamp.type
    assert isinstance(column_type, DateTime)
    assert column_type.timezone is True


# --- assessment / human_decision -- [M5-03] --------------------------------


@pytest.mark.unit
def test_assessment_table_name_and_primary_key() -> None:
    assert AssessmentRow.__tablename__ == "assessment"
    assert [c.name for c in _ASSESSMENT_TABLE.primary_key.columns] == ["assessment_id"]


@pytest.mark.unit
def test_assessment_mirrors_every_record_field() -> None:
    # The aggregate's fields, minus the nested `decision` (its own table) and
    # the identity that is the PK, must each have a column.
    expected = {
        "assessment_id",
        "claim_id",
        "verdict",
        "reasoning",
        "recommended_action",
        "confidence",
        "context_sufficient",
        "clarification_exhausted",
        "missing_information",
        "citations",
        "consistency_flags",
        "status",
        "created_at",
    }
    assert expected == set(_ASSESSMENT_TABLE.c.keys())


@pytest.mark.unit
def test_assessment_context_sufficient_is_the_only_nullable_column() -> None:
    # `context_sufficient` is genuinely tri-state; everything else is required.
    nullable = {c.name for c in _ASSESSMENT_TABLE.c if c.nullable}
    assert nullable == {"context_sufficient"}


@pytest.mark.unit
def test_assessment_check_constraints_match_the_domain_enums() -> None:
    assert _check_constraint_values(
        "ck_assessment_verdict_valid", _ASSESSMENT_TABLE
    ) == {member.value for member in Verdict}
    assert _check_constraint_values(
        "ck_assessment_status_valid", _ASSESSMENT_TABLE
    ) == {member.value for member in AssessmentStatus}


@pytest.mark.unit
def test_assessment_indexes_the_columns_list_queries_by() -> None:
    assert {str(index.name) for index in _ASSESSMENT_TABLE.indexes} == {
        "ix_assessment_claim_id",
        "ix_assessment_status",
        "ix_assessment_created_at",
    }


@pytest.mark.unit
def test_assessment_created_at_is_timezone_aware() -> None:
    column_type = _ASSESSMENT_TABLE.c.created_at.type
    assert isinstance(column_type, DateTime)
    assert column_type.timezone is True


@pytest.mark.unit
def test_human_decision_is_keyed_and_foreign_keyed_to_its_assessment() -> None:
    assert HumanDecisionRow.__tablename__ == "human_decision"
    assert [c.name for c in _DECISION_TABLE.primary_key.columns] == ["assessment_id"]
    foreign_keys = list(_DECISION_TABLE.c.assessment_id.foreign_keys)
    assert len(foreign_keys) == 1
    assert foreign_keys[0].column.table.name == "assessment"


@pytest.mark.unit
def test_human_decision_check_constraint_matches_the_decision_enum() -> None:
    assert _check_constraint_values(
        "ck_human_decision_decision_valid", _DECISION_TABLE
    ) == {member.value for member in DecisionOutcome}


@pytest.mark.unit
def test_human_decision_pairs_edited_assessment_with_the_edit_outcome() -> None:
    # Mirrors `HumanDecision._check_edit_carries_a_revision`.
    names = {
        str(c.name)
        for c in _DECISION_TABLE.constraints
        if isinstance(c, CheckConstraint)
    }
    assert "ck_human_decision_edited_assessment_iff_edit" in names
    assert _DECISION_TABLE.c.edited_assessment.nullable is True


@pytest.mark.unit
def test_human_decision_decided_at_is_timezone_aware() -> None:
    column_type = _DECISION_TABLE.c.decided_at.type
    assert isinstance(column_type, DateTime)
    assert column_type.timezone is True


# --------------------------------------------------------------------------- #
# assessment_job -- [M5-05]
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_assessment_job_table_name_and_primary_key() -> None:
    assert AssessmentJobRow.__tablename__ == "assessment_job"
    assert [c.name for c in _JOB_TABLE.primary_key.columns] == ["assessment_id"]


@pytest.mark.unit
def test_assessment_job_status_check_matches_the_job_status_enum() -> None:
    assert _check_constraint_values("ck_assessment_job_status_valid", _JOB_TABLE) == {
        member.value for member in JobStatus
    }


@pytest.mark.unit
def test_assessment_job_has_no_foreign_key_to_assessment() -> None:
    # The job row exists before any assessment does, and a failed run keeps its
    # job row without ever producing one.
    assert list(_JOB_TABLE.foreign_keys) == []


@pytest.mark.unit
def test_assessment_job_timestamps_are_timezone_aware() -> None:
    for name in ("submitted_at", "created_at", "updated_at"):
        column_type = _JOB_TABLE.c[name].type
        assert isinstance(column_type, DateTime)
        assert column_type.timezone is True


@pytest.mark.unit
def test_assessment_job_indexes_status_and_claim_id() -> None:
    assert {str(index.name) for index in _JOB_TABLE.indexes} == {
        "ix_assessment_job_status",
        "ix_assessment_job_claim_id",
    }
