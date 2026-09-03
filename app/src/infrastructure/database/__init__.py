"""Database adapters.

SQLAlchemy models, repositories, session management, and the pgvector
integration.
"""

from infrastructure.database.assessment_mapper import (
    record_to_rows,
    rows_to_record,
)
from infrastructure.database.assessment_repository import (
    SqlAlchemyAssessmentRepository,
)
from infrastructure.database.audit_repository import (
    append_audit_entries,
    append_audit_events,
)
from infrastructure.database.audit_trail_reader import SqlAlchemyAuditTrailReader
from infrastructure.database.audit_trail_writer import SqlAlchemyAuditTrailWriter
from infrastructure.database.base import Base
from infrastructure.database.chunk_repository import (
    assert_chunk_table_ready,
    fetch_chunks_missing_embedding,
    upsert_chunks,
    write_chunk_embeddings,
)
from infrastructure.database.clause_repository import SqlAlchemyClauseRepository
from infrastructure.database.graph_audit_sink import SqlAlchemyAuditTrailSink
from infrastructure.database.models import (
    AssessmentRow,
    AuditEventRow,
    ChunkRow,
    HumanDecisionRow,
)
from infrastructure.database.session import (
    create_engine_from_database_url,
    create_engine_from_settings,
    create_session_factory,
    is_database_reachable,
)
from infrastructure.database.unit_of_work import (
    SqlAlchemyUnitOfWork,
    sqlalchemy_unit_of_work_factory,
)

__all__ = [
    "AssessmentRow",
    "AuditEventRow",
    "Base",
    "ChunkRow",
    "HumanDecisionRow",
    "SqlAlchemyAssessmentRepository",
    "SqlAlchemyAuditTrailReader",
    "SqlAlchemyAuditTrailSink",
    "SqlAlchemyAuditTrailWriter",
    "SqlAlchemyClauseRepository",
    "SqlAlchemyUnitOfWork",
    "append_audit_entries",
    "append_audit_events",
    "assert_chunk_table_ready",
    "create_engine_from_database_url",
    "create_engine_from_settings",
    "create_session_factory",
    "fetch_chunks_missing_embedding",
    "is_database_reachable",
    "record_to_rows",
    "rows_to_record",
    "sqlalchemy_unit_of_work_factory",
    "upsert_chunks",
    "write_chunk_embeddings",
]
