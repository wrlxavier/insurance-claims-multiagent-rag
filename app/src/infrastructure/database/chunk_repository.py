"""Idempotent write path for the chunk table -- [M3-02].

``chunk_id`` is deterministic upstream ([M3-01]/[M1-07]), so writing the chunk
corpus is an upsert on that key: a re-run over the same corpus neither
duplicates rows nor requires a wipe.
"""

from collections.abc import Iterable
from itertools import batched

from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from infrastructure.database.models import ChunkRow
from infrastructure.rag.chunk_schema import ChunkRecord

# Every column except the conflict key is refreshed from the incoming row, so a
# changed chunk (new text, new type) overwrites the stored one in place.
_UPDATE_COLUMNS = tuple(
    column.name for column in ChunkRow.__table__.columns if column.name != "chunk_id"
)

# 23 columns per row; Postgres caps a statement at 65535 bound parameters, so
# keep batches well under 65535 / 23 ~= 2800.
_DEFAULT_BATCH_SIZE = 1000


def _row_values(record: ChunkRecord) -> dict[str, object]:
    """Map a [ChunkRecord] to a ``chunk`` row (``text`` -> ``embedded_text``)."""
    data: dict[str, object] = record.model_dump(mode="json")
    data["embedded_text"] = data.pop("text")
    return data


def upsert_chunks(
    session: Session,
    records: Iterable[ChunkRecord],
    *,
    batch_size: int = _DEFAULT_BATCH_SIZE,
) -> int:
    """Insert or update chunk rows by ``chunk_id``. Return the number written.

    The caller owns the transaction -- this flushes each batch but never
    commits.
    """
    statement = pg_insert(ChunkRow)
    statement = statement.on_conflict_do_update(
        index_elements=["chunk_id"],
        set_={name: statement.excluded[name] for name in _UPDATE_COLUMNS},
    )

    written = 0
    for batch in batched(records, batch_size):
        session.execute(statement, [_row_values(record) for record in batch])
        session.flush()
        written += len(batch)
    return written
