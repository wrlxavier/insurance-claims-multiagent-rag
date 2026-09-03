"""Database adapters.

SQLAlchemy models, repositories, session management, and the pgvector
integration.
"""

from infrastructure.database.audit_repository import append_audit_events
from infrastructure.database.base import Base
from infrastructure.database.chunk_repository import (
    assert_chunk_table_ready,
    fetch_chunks_missing_embedding,
    upsert_chunks,
    write_chunk_embeddings,
)
from infrastructure.database.graph_audit_sink import SqlAlchemyAuditTrailSink
from infrastructure.database.models import AuditEventRow, ChunkRow
from infrastructure.database.session import (
    create_engine_from_database_url,
    create_engine_from_settings,
    create_session_factory,
    is_database_reachable,
)

__all__ = [
    "AuditEventRow",
    "Base",
    "ChunkRow",
    "SqlAlchemyAuditTrailSink",
    "append_audit_events",
    "assert_chunk_table_ready",
    "create_engine_from_database_url",
    "create_engine_from_settings",
    "create_session_factory",
    "fetch_chunks_missing_embedding",
    "is_database_reachable",
    "upsert_chunks",
    "write_chunk_embeddings",
]
