"""Write paths for the chunk table -- [M3-02].

Two sinks, deliberately separate:

* ``upsert_chunks`` -- the metadata write path. ``chunk_id`` is deterministic
  upstream ([M3-01]/[M1-07]), so writing the chunk corpus is an upsert on that
  key: a re-run over the same corpus neither duplicates rows nor requires a
  wipe. It never touches ``embedding`` (see ``_UPDATE_COLUMNS``), so refreshing
  chunk metadata does not wipe vectors already computed.
* ``fetch_chunks_missing_embedding`` / ``write_chunk_embeddings`` -- the
  embedding pipeline's read cursor and vector sink. The pipeline owns the
  ``embedding`` column exclusively.
"""

from collections.abc import Iterable, Mapping, Sequence
from itertools import batched

from sqlalchemy import select, text, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from infrastructure.database.models import ChunkRow
from infrastructure.rag.chunk_schema import ChunkRecord

# Every column except the conflict key and ``embedding`` is refreshed from the
# incoming row, so a changed chunk (new text, new type) overwrites the stored
# one in place. ``embedding`` is left out on purpose: a ``ChunkRecord`` carries
# no vector, and including it here would set ``embedding = NULL`` on every
# metadata re-run and silently discard the whole embedded corpus. The embedding
# pipeline is the only writer of that column.
_SKIP_ON_UPDATE = frozenset({"chunk_id", "embedding"})
_UPDATE_COLUMNS = tuple(
    column.name
    for column in ChunkRow.__table__.columns
    if column.name not in _SKIP_ON_UPDATE
)

# 23 non-vector columns per row; Postgres caps a statement at 65535 bound
# parameters, so keep batches well under 65535 / 23 ~= 2800.
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


def assert_chunk_table_ready(session: Session) -> None:
    """Fail loudly with the fix command if the ``chunk`` table is not migrated.

    Checks for the ``embedding`` column specifically, so it also catches a
    database migrated to before ``20260827_03``. Shared by every script that
    reads or writes ``chunk`` (``load_chunks``, ``embed_chunks``,
    ``benchmark_ann_index``).
    """
    ready = session.execute(
        text(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_name = 'chunk' AND column_name = 'embedding'"
        )
    ).first()
    if ready is None:
        raise RuntimeError(
            "`chunk` table (with the `embedding` column) not found. Run "
            "`make migrate` (or `DATABASE_URL=$TEST_DATABASE_URL make migrate` "
            "for the test database)."
        )


def fetch_chunks_missing_embedding(session: Session) -> list[tuple[str, str]]:
    """Return ``(chunk_id, embedded_text)`` for every chunk with no vector yet.

    This is the embedding pipeline's resumable cursor: an interrupted run leaves
    the chunks it did not reach at ``embedding IS NULL``, and re-running picks up
    exactly that remainder. Ordered by ``chunk_id`` for a deterministic pass.
    The whole pending set is small (one row per un-embedded chunk, id + text),
    so it is materialised here rather than streamed -- the pipeline commits
    between batches, which would invalidate an open server-side cursor.
    """
    rows = session.execute(
        select(ChunkRow.chunk_id, ChunkRow.embedded_text)
        .where(ChunkRow.embedding.is_(None))
        .order_by(ChunkRow.chunk_id)
    ).all()
    return [(chunk_id, embedded_text) for chunk_id, embedded_text in rows]


def write_chunk_embeddings(
    session: Session, vectors: Mapping[str, Sequence[float]]
) -> int:
    """Write one vector per ``chunk_id`` into ``chunk.embedding``. Return the count.

    A bulk UPDATE keyed on the primary key -- every ``chunk_id`` must already
    exist (the pipeline reads them from :func:`fetch_chunks_missing_embedding`).
    Flushes but never commits; the pipeline commits per batch so an interrupted
    run keeps every batch it completed.
    """
    if not vectors:
        return 0
    session.execute(
        update(ChunkRow),
        [
            {"chunk_id": chunk_id, "embedding": list(vector)}
            for chunk_id, vector in vectors.items()
        ],
    )
    session.flush()
    return len(vectors)
