"""SQLAlchemy ORM models -- [M3-02], [M4-09], [M5-03].

``ChunkRow`` is the persistence representation of [domain.chunk.Chunk] /
[infrastructure.rag.chunk_schema.ChunkRecord]: the chunk corpus indexed in
Postgres for retrieval. It is a third representation on purpose -- the domain
dataclass stays stdlib-only (pydantic and sqlalchemy are forbidden imports in
domain/application, see tests/architecture/test_layer_boundaries.py), the
pydantic ``ChunkRecord`` is the ``build/`` serialization row, and this is the
table.

Scope note: the ``embedding`` column is the ``halfvec(768)`` vector the [M3-02]
embedding pipeline fills. Its HNSW ANN index is defined in
``infrastructure.rag.ann_index`` -- deliberately not a migration; the
ANN-vs-exact measurement and the verdict are in docs/EMBEDDINGS.md.

``AuditEventRow`` ([M4-09]) is the same idea applied to the graph's audit trail:
the persistence representation of [infrastructure.graph.state.AuditEvent].

``AssessmentRow`` / ``HumanDecisionRow`` ([M5-03]) are the persistence
representation of the servable aggregate
[application.assessment_record.AssessmentRecord] and the analyst's
[domain.human_decision.HumanDecision] recorded beside it. ``citations`` and
``consistency_flags`` are ``JSONB`` rather than child tables: they are frozen
value-object tuples read back whole with the record and never queried by field
(the same call as ``AuditEventRow.payload``). The domain <-> row mapper lives in
``infrastructure.database.assessment_mapper``.
"""

from collections.abc import Iterable
from datetime import datetime

from pgvector.sqlalchemy import HALFVEC
from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import Mapped, mapped_column

from application.assessment_job import JobStatus
from application.assessment_record import AssessmentStatus
from domain.chunk import ChunkRule
from domain.clause_classification import ClauseType, TypeSource
from domain.human_decision import DecisionOutcome
from domain.verdict import Verdict
from infrastructure.database.base import Base
from infrastructure.rag.embedding_config import EMBEDDING_DIMENSIONS

_SOURCE_VALUES = ("text", "ocr")


def _in_clause(column: str, values: Iterable[str]) -> str:
    """Render ``column IN ('a', 'b', ...)`` for a CHECK constraint."""
    rendered = ", ".join(f"'{value}'" for value in values)
    return f"{column} IN ({rendered})"


class ChunkRow(Base):
    """One chunk row: clause text plus the metadata [M3-04] filters by.

    Column names mirror [infrastructure.rag.chunk_schema.ChunkRecord] field for
    field (except ``text`` -> ``embedded_text``), so the write path in
    [infrastructure.database.chunk_repository] is a direct mapping and "carry
    the full [M1-05] provenance" is auditable column by column.
    """

    __tablename__ = "chunk"

    # `chunk_id` is deterministic upstream ([M3-01]/[M1-07]): `clause_id` for a
    # one-chunk clause, `f"{clause_id}#{index}"` for a split, where
    # `clause_id = f"{document_id}:{path}"`. Same input, same id -- so it is the
    # natural key, and the write path upserts on it (a re-run neither
    # duplicates nor needs a wipe).
    chunk_id: Mapped[str] = mapped_column(Text, primary_key=True)

    schema_version: Mapped[str] = mapped_column(Text, nullable=False)
    document_id: Mapped[str] = mapped_column(Text, nullable=False)
    clause_id: Mapped[str] = mapped_column(Text, nullable=False)
    source_clause_ids: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False)

    # "" when the anchor clause has no ancestors -- a real value, never NULL.
    parent_path: Mapped[str] = mapped_column(Text, nullable=False)
    # The string the embedding model sees (ancestor breadcrumb prepended).
    embedded_text: Mapped[str] = mapped_column(Text, nullable=False)
    # The same clause with only the injected breadcrumb removed -- [M4-01]'s
    # citation type needs a quoted excerpt, which is not `embedded_text`.
    display_text: Mapped[str] = mapped_column(Text, nullable=False)
    # Measures `embedded_text`.
    char_count: Mapped[int] = mapped_column(Integer, nullable=False)

    rule: Mapped[str] = mapped_column(Text, nullable=False)
    clause_type: Mapped[str] = mapped_column(Text, nullable=False)
    type_source: Mapped[str] = mapped_column(Text, nullable=False)
    source: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Genuinely nullable: no server_default, no sentinel. A strict M3-04 filter
    # `WHERE bundle_section = :x` must silently (and testably in SQL) exclude
    # unknown-bundle chunks -- see the [M1-06] cross-note in [M3-04].
    bundle_section: Mapped[str | None] = mapped_column(Text, nullable=True)

    susep_process: Mapped[str] = mapped_column(Text, nullable=False)
    # Not indexed: M3-04 filters insurers by CNPJ, never by name (HDI Seguros
    # vs HDI Global share a brand but are different legal entities).
    insurer: Mapped[str] = mapped_column(Text, nullable=False)
    cnpj: Mapped[str] = mapped_column(Text, nullable=False)
    product_line: Mapped[str] = mapped_column(Text, nullable=False)
    indemnity_regime: Mapped[str] = mapped_column(Text, nullable=False)
    filing_year: Mapped[str] = mapped_column(Text, nullable=False)

    # The dense-retrieval vector for `embedded_text`, per the pinned model
    # contract in `infrastructure.rag.embedding_config` (768-dim, cosine,
    # L2-normalised). `halfvec` half-precision storage follows [M0-08]'s
    # decision. Genuinely nullable: an un-embedded chunk is `NULL`, and
    # `WHERE embedding IS NULL` is the embedding pipeline's resumable cursor.
    # `upsert_chunks` (the metadata write path) deliberately never writes this
    # column -- see `chunk_repository._UPDATE_COLUMNS`. The HNSW ANN index over
    # it is defined in `infrastructure.rag.ann_index`, not a migration -- see
    # its docstring and docs/EMBEDDINGS.md.
    embedding: Mapped[list[float] | None] = mapped_column(
        HALFVEC(EMBEDDING_DIMENSIONS), nullable=True
    )

    __table_args__ = (
        CheckConstraint(
            _in_clause("rule", (member.value for member in ChunkRule)),
            name="rule_valid",
        ),
        CheckConstraint(
            _in_clause("clause_type", (member.value for member in ClauseType)),
            name="clause_type_valid",
        ),
        CheckConstraint(
            _in_clause("type_source", (member.value for member in TypeSource)),
            name="type_source_valid",
        ),
        CheckConstraint(
            _in_clause("source", _SOURCE_VALUES),
            name="source_valid",
        ),
        # One index per field M3-04 filters by, plus the composite for its
        # default path (SUSEP process + insurer CNPJ together). Named
        # explicitly rather than via `index=True` so the migration and the
        # model agree on the exact names with no naming-convention ambiguity.
        # All forward-looking: at ~4,900 chunks Postgres seq-scans in well
        # under a millisecond regardless.
        Index("ix_chunk_clause_type", "clause_type"),
        Index("ix_chunk_bundle_section", "bundle_section"),
        Index("ix_chunk_susep_process", "susep_process"),
        Index("ix_chunk_cnpj", "cnpj"),
        Index("ix_chunk_product_line", "product_line"),
        Index("ix_chunk_susep_process_cnpj", "susep_process", "cnpj"),
    )


class AuditEventRow(Base):
    """One entry of a graph run's audit trail, as a durable row -- [M4-09].

    The persistence representation of
    [infrastructure.graph.state.AuditEvent], the same way ``ChunkRow`` is the
    persistence representation of a ``ChunkRecord``: column per field, so the
    trail is auditable in SQL without going through LangGraph's checkpoint serde.

    Three fields have no counterpart in ``AuditEvent`` and are added here:

    * ``thread_id`` / ``sequence`` -- the composite primary key, and the whole
      idempotency story. The checkpoint node that writes this trail runs again
      every time its thread is resumed ([M4-09]'s re-execution semantics), so the
      write has to be safe to repeat; position within a thread's trail is
      deterministic, which makes an ``ON CONFLICT DO NOTHING`` insert enough and
      spares the table an invented event id.
    * ``claim_id`` -- what a human searches by. Indexed; ``thread_id`` is already
      covered by the primary key.
    * ``payload`` -- optional JSON detail the flat event has no field for. The
      human-review event fills it with the analyst's whole ``HumanDecision``.

    ``AuditEvent.token_usage`` is flattened into three nullable integer columns
    rather than nested: a single-row-per-event table keeps the trail queryable
    with plain aggregates, and ``TokenUsage`` is a closed three-field record.

    Scope note: this table is [M4-09]'s, moved forward from [M5-03] because
    [M4-09]'s DoD requires the trail to be durable and separate from graph state.
    [M5-03] keeps the domain tables (assessments, decisions) and adds the
    database-level append-only enforcement its own DoD asks for -- there is no
    update path in [infrastructure.database.audit_repository] in the meantime.
    """

    __tablename__ = "audit_event"

    thread_id: Mapped[str] = mapped_column(Text, primary_key=True)
    sequence: Mapped[int] = mapped_column(Integer, primary_key=True)

    claim_id: Mapped[str] = mapped_column(Text, nullable=False)

    # Mirrors `AuditEvent` field for field from here down.
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    node: Mapped[str] = mapped_column(Text, nullable=False)
    action: Mapped[str] = mapped_column(Text, nullable=False)
    model: Mapped[str | None] = mapped_column(Text, nullable=True)
    model_version: Mapped[str | None] = mapped_column(Text, nullable=True)
    input_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    output_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    node_input: Mapped[str | None] = mapped_column(Text, nullable=True)

    payload: Mapped[dict[str, object] | None] = mapped_column(JSONB, nullable=True)

    __table_args__ = (Index("ix_audit_event_claim_id", "claim_id"),)


class AssessmentRow(Base):
    """One claim's assessment across its whole lifecycle -- [M5-03].

    The persistence representation of
    [application.assessment_record.AssessmentRecord] (the servable aggregate,
    *not* the grounded [domain.assessment.Assessment] -- an abstain outcome has
    no citations and still has to be stored and served). Column per field, so a
    reader can query the trail in SQL; the domain <-> row mapper is in
    [infrastructure.database.assessment_mapper].

    ``citations`` and ``consistency_flags`` are ``JSONB`` arrays of objects, not
    child tables: both are frozen value-object tuples with no identity of their
    own, always loaded whole with the record and never filtered by field -- the
    same reasoning as ``AuditEventRow.payload``. ``missing_information`` is a
    plain ``TEXT[]`` (like ``ChunkRow.source_clause_ids``).

    The enum-valued columns (``verdict``, ``status``) are ``TEXT`` + a named
    ``CHECK``, matching ``ChunkRow``'s precedent -- no native PG enums in this
    project. ``context_sufficient`` is genuinely tri-state
    (``True``/``False``/``None`` -- retrieval succeeded / gated / never ran).
    """

    __tablename__ = "assessment"

    assessment_id: Mapped[str] = mapped_column(Text, primary_key=True)
    claim_id: Mapped[str] = mapped_column(Text, nullable=False)

    verdict: Mapped[str] = mapped_column(Text, nullable=False)
    reasoning: Mapped[str] = mapped_column(Text, nullable=False)
    recommended_action: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)

    # Tri-state on purpose -- see the class docstring.
    context_sufficient: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    clarification_exhausted: Mapped[bool] = mapped_column(Boolean, nullable=False)
    missing_information: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False)

    citations: Mapped[list[dict[str, object]]] = mapped_column(JSONB, nullable=False)
    consistency_flags: Mapped[list[dict[str, object]]] = mapped_column(
        JSONB, nullable=False
    )

    status: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )

    __table_args__ = (
        CheckConstraint(
            _in_clause("verdict", (member.value for member in Verdict)),
            name="verdict_valid",
        ),
        CheckConstraint(
            _in_clause("status", (member.value for member in AssessmentStatus)),
            name="status_valid",
        ),
        # `claim_id` -- every assessment of one claim ([List] filter, and what a
        # human searches by). `status` -- the awaiting-review queue. `created_at`
        # -- `AssessmentRepository.list` orders by it (newest first).
        Index("ix_assessment_claim_id", "claim_id"),
        Index("ix_assessment_status", "status"),
        Index("ix_assessment_created_at", "created_at"),
    )


class AssessmentJobRow(Base):
    """One queued assessment run's lifecycle -- [M5-05].

    The persistence representation of [application.assessment_job.AssessmentJob]:
    the run state a caller polls (``pending`` -> ``running`` -> ``succeeded`` /
    ``failed``) plus what a worker needs to rebuild the domain ``Claim`` after a
    redelivery or a retry (``raw_text``, ``policy_ref``, and ``submitted_at`` --
    stamped once at submission so a retry keeps the same loss-date baseline).

    ``failure`` is ``JSONB`` (``kind`` / ``error_type`` / ``message`` /
    ``failed_at``): a frozen value object read whole with the job, never queried
    by field -- the same call as ``AuditEventRow.payload``. ``status`` is
    ``TEXT`` + a named CHECK from ``JobStatus`` (chunk/assessment precedent -- no
    native PG enums here). No FK to ``assessment``: the job row is created before
    any assessment exists, and outlives a failed run that never produces one.
    """

    __tablename__ = "assessment_job"

    assessment_id: Mapped[str] = mapped_column(Text, primary_key=True)
    claim_id: Mapped[str] = mapped_column(Text, nullable=False)

    raw_text: Mapped[str] = mapped_column(Text, nullable=False)
    # The SUSEP process the claim is filed against, canonical form -- or NULL when
    # the submission carried none (intake extracts it from the narrative instead).
    policy_ref: Mapped[str | None] = mapped_column(Text, nullable=True)
    submitted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )

    status: Mapped[str] = mapped_column(Text, nullable=False)
    attempts: Mapped[int] = mapped_column(Integer, nullable=False)
    failure: Mapped[dict[str, object] | None] = mapped_column(JSONB, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )

    __table_args__ = (
        CheckConstraint(
            _in_clause("status", (member.value for member in JobStatus)),
            name="status_valid",
        ),
        # `status` -- the worker scans for `pending`; `claim_id` -- what a human
        # searches by (every run of one claim). `created_at` is not indexed:
        # unlike `assessment`, jobs are not listed newest-first through the API.
        Index("ix_assessment_job_status", "status"),
        Index("ix_assessment_job_claim_id", "claim_id"),
    )


class HumanDecisionRow(Base):
    """The analyst's decision on one assessment -- [M5-03].

    The persistence representation of [domain.human_decision.HumanDecision]: one
    row per *settled* assessment, recorded beside the system's opinion, never
    over it. The primary key is also a foreign key to ``assessment`` -- a
    decision "always references the assessment it acted on" is the M5-01
    invariant, made structural here.

    ``edited_assessment`` is the analyst's revised [domain.assessment.Assessment]
    as ``JSONB``, present exactly when ``decision = 'edit'`` -- a ``CHECK``
    enforces the biconditional, mirroring ``HumanDecision``'s own validator.
    """

    __tablename__ = "human_decision"

    assessment_id: Mapped[str] = mapped_column(
        Text,
        ForeignKey("assessment.assessment_id"),
        primary_key=True,
    )
    decision: Mapped[str] = mapped_column(Text, nullable=False)
    decided_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    notes: Mapped[str] = mapped_column(Text, nullable=False)
    # `none_as_null`: a Python `None` must become SQL NULL, not the JSON value
    # `null` -- otherwise `edited_assessment IS NOT NULL` is true for a
    # non-`edit` decision and the CHECK below fires.
    edited_assessment: Mapped[dict[str, object] | None] = mapped_column(
        JSONB(none_as_null=True), nullable=True
    )

    __table_args__ = (
        CheckConstraint(
            _in_clause("decision", (member.value for member in DecisionOutcome)),
            name="decision_valid",
        ),
        # `edit` carries a revision; nothing else may -- the same rule
        # `HumanDecision._check_edit_carries_a_revision` enforces in the domain.
        CheckConstraint(
            "(decision = 'edit') = (edited_assessment IS NOT NULL)",
            name="edited_assessment_iff_edit",
        ),
    )
