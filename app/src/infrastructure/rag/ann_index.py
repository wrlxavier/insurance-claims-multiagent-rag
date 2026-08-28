"""The HNSW ANN index over ``chunk.embedding`` and its search-time GUCs -- [M3-02].

Definition only -- deliberately **not** an Alembic migration. An ANN index is a
retrieval-tuning artifact, not schema: exact ``<=>`` cosine search works without
it (``docs/EMBEDDINGS.md`` measures the difference and records the verdict), its
``m`` / ``ef_construction`` are [M3-08]'s to tune, and the autouse
``migrated_database`` fixture in ``tests/integration/conftest.py`` would rebuild
it on every integration test. [M3-08]'s ``make build-index`` calls
:func:`create_hnsw_index` (iff the recorded verdict says the index earns its
place); a retriever calls :func:`apply_ann_search_gucs` only on a query that
routes through the index. [M3-04]'s dense retriever does **not** -- its default
path is exact ``<=>`` over the pre-filtered partition (``docs/EMBEDDINGS.md``,
"Filtered search and the fewer-than-``k`` question"), so this helper is
currently reached only by ``benchmark_ann_index`` and [M3-08]'s future matrix.

The parameters are module constants, not ``.env`` knobs: ``m`` / ``ef_construction``
change the index (and therefore any published Recall@k), and ``ef_search`` /
``iterative_scan`` change what a filtered search *returns*. Under [M1-09]'s
per-constant rule they are experimental design, not operational knobs -- see the
per-constant decision table in ``docs/EMBEDDINGS.md``. (``EMBEDDING_BATCH_SIZE``
was the one M3-02 constant that rule moved to ``.env``.)
"""

from typing import Literal, get_args

from sqlalchemy import text
from sqlalchemy.orm import Session

# Follows the repo `ix_chunk_*` index-naming convention. Never declared on
# `ChunkRow.__table_args__` -- it is created imperatively here, so
# `test_models.py::test_embedding_column_is_not_indexed_yet` stays true.
INDEX_NAME = "ix_chunk_embedding_hnsw"

# pgvector's own defaults. Kept explicit so the benchmark records the exact
# configuration it measured and [M3-08] has one place to tune.
HNSW_M = 16
HNSW_EF_CONSTRUCTION = 64
HNSW_EF_SEARCH = 40

IterativeScan = Literal["off", "strict_order", "relaxed_order"]

# `strict_order` is the documented default for the filtered retrieval path: at
# ~4,540 rows the candidate set is far inside `hnsw.max_scan_tuples` (20000), so
# a filtered search keeps scanning until it has k results *in exact distance
# order* at negligible cost -- and [M3-04]'s RRF fusion / [M3-08]'s ranked
# metrics must not carry approximate ordering as a confound. See docs/EMBEDDINGS.md.
HNSW_ITERATIVE_SCAN: IterativeScan = "strict_order"

_CREATE_INDEX = text(
    f"CREATE INDEX {INDEX_NAME} ON chunk "
    f"USING hnsw (embedding halfvec_cosine_ops) "
    f"WITH (m = {HNSW_M}, ef_construction = {HNSW_EF_CONSTRUCTION})"
)
_DROP_INDEX = text(f"DROP INDEX IF EXISTS {INDEX_NAME}")


def create_hnsw_index(session: Session) -> None:
    """Create the HNSW index over ``chunk.embedding``. The caller owns the transaction.

    Matches the pinned contract in ``embedding_config`` -- ``halfvec(768)``
    storage, ``halfvec_cosine_ops`` for the ``<=>`` cosine operator.
    """
    session.execute(_CREATE_INDEX)


def drop_hnsw_index(session: Session) -> None:
    """Drop the HNSW index if it exists. The caller owns the transaction."""
    session.execute(_DROP_INDEX)


def apply_ann_search_gucs(
    session: Session,
    *,
    ef_search: int = HNSW_EF_SEARCH,
    iterative_scan: IterativeScan = HNSW_ITERATIVE_SCAN,
) -> None:
    """``SET LOCAL`` the pgvector HNSW search knobs for the current transaction.

    Without ``iterative_scan``, a metadata pre-filter combined with an HNSW scan
    can return fewer than ``k`` rows even when ``>= k`` rows match the filter:
    the executor filters the ``ef_search`` candidate list *after* the index
    scan. ``strict_order`` makes the index keep scanning until ``k`` post-filter
    matches are found. See ``docs/EMBEDDINGS.md`` for the full case analysis and
    ``tests/integration/test_ann_index.py`` for the committed proof.

    ``SET LOCAL`` does not take bind parameters, so the values are interpolated;
    ``ef_search`` is coerced to ``int`` and ``iterative_scan`` is validated
    against the ``Literal`` members first.
    """
    if iterative_scan not in get_args(IterativeScan):
        raise ValueError(f"invalid iterative_scan: {iterative_scan!r}")
    session.execute(text(f"SET LOCAL hnsw.ef_search = {int(ef_search)}"))
    session.execute(text(f"SET LOCAL hnsw.iterative_scan = '{iterative_scan}'"))
