"""Unit tests for the HNSW ANN index helper -- [M3-02].

No database: a recording fake session captures the SQL each helper emits. The
against-a-real-Postgres behaviour (the index builds, the planner uses it, a
filtered scan can under-return and ``strict_order`` restores ``k``) is in
``tests/integration/test_ann_index.py``.
"""

from typing import Any

import pytest

from infrastructure.rag.ann_index import (
    HNSW_EF_CONSTRUCTION,
    HNSW_EF_SEARCH,
    HNSW_M,
    INDEX_NAME,
    apply_ann_search_gucs,
    create_hnsw_index,
    drop_hnsw_index,
)


class _RecordingSession:
    """Records the string form of every statement passed to ``execute``."""

    def __init__(self) -> None:
        self.statements: list[str] = []

    def execute(self, clause: Any, *args: Any, **kwargs: Any) -> None:
        self.statements.append(str(clause))


@pytest.mark.unit
def test_index_name_follows_the_repo_convention() -> None:
    assert INDEX_NAME.startswith("ix_chunk_")


@pytest.mark.unit
def test_create_hnsw_index_emits_the_expected_ddl() -> None:
    session = _RecordingSession()

    create_hnsw_index(session)  # type: ignore[arg-type]

    assert len(session.statements) == 1
    ddl = session.statements[0]
    assert f"CREATE INDEX {INDEX_NAME} ON chunk" in ddl
    assert "USING hnsw (embedding halfvec_cosine_ops)" in ddl
    assert f"m = {HNSW_M}" in ddl
    assert f"ef_construction = {HNSW_EF_CONSTRUCTION}" in ddl


@pytest.mark.unit
def test_drop_hnsw_index_is_idempotent_ddl() -> None:
    session = _RecordingSession()

    drop_hnsw_index(session)  # type: ignore[arg-type]

    assert session.statements == [f"DROP INDEX IF EXISTS {INDEX_NAME}"]


@pytest.mark.unit
def test_apply_ann_search_gucs_emits_both_set_local_statements() -> None:
    session = _RecordingSession()

    apply_ann_search_gucs(session, ef_search=25, iterative_scan="relaxed_order")  # type: ignore[arg-type]

    assert session.statements == [
        "SET LOCAL hnsw.ef_search = 25",
        "SET LOCAL hnsw.iterative_scan = 'relaxed_order'",
    ]


@pytest.mark.unit
def test_apply_ann_search_gucs_defaults_to_strict_order_and_the_pinned_ef_search() -> (
    None
):
    session = _RecordingSession()

    apply_ann_search_gucs(session)  # type: ignore[arg-type]

    assert session.statements == [
        f"SET LOCAL hnsw.ef_search = {HNSW_EF_SEARCH}",
        "SET LOCAL hnsw.iterative_scan = 'strict_order'",
    ]


@pytest.mark.unit
def test_apply_ann_search_gucs_rejects_an_unknown_iterative_scan_mode() -> None:
    session = _RecordingSession()

    with pytest.raises(ValueError, match="invalid iterative_scan"):
        apply_ann_search_gucs(session, iterative_scan="turbo")  # type: ignore[arg-type]

    assert session.statements == []  # nothing executed before the guard fires
