from pathlib import Path

import pytest

from domain.clause_tree import (
    Clause,
    ClauseTree,
    ClauseTreeReport,
    ClauseTreeWarning,
    HeadingConvention,
)
from infrastructure.parsing.clause_tree_caching import (
    clause_tree_cache_path,
    compute_clause_tree_cache_key,
    read_clause_tree_cache,
    write_clause_tree_cache,
)


@pytest.mark.unit
def test_compute_clause_tree_cache_key_is_deterministic() -> None:
    assert compute_clause_tree_cache_key("v1") == compute_clause_tree_cache_key("v1")


@pytest.mark.unit
def test_compute_clause_tree_cache_key_changes_with_version() -> None:
    assert compute_clause_tree_cache_key("v1") != compute_clause_tree_cache_key("v2")


@pytest.mark.unit
def test_clause_tree_cache_path_names_file_by_document_id_and_key() -> None:
    path = clause_tree_cache_path("10", "abc123")

    assert path.name == "10__abc123.parquet"


def _tree() -> ClauseTree:
    root = Clause(
        document_id="17",
        clause_id="17:1",
        path="1",
        numbering_label="1",
        title="1. OBJETO DO SEGURO",
        convention=HeadingConvention.NUMBERED_DECIMAL,
        depth=1,
        parent_id=None,
        child_ids=("17:2",),
        content_lines=("Corpo do objeto.",),
        page_start=1,
        page_end=2,
        bundle_section=None,
        bundle_confidence=None,
        is_depth_anomaly=False,
    )
    child = Clause(
        document_id="17",
        clause_id="17:2",
        path="1/1.1",
        numbering_label="1.1",
        title="1.1 Âmbito Geográfico",
        convention=HeadingConvention.NUMBERED_DECIMAL,
        depth=2,
        parent_id="17:1",
        child_ids=(),
        content_lines=(),
        page_start=2,
        page_end=2,
        bundle_section="1. OBJETO DO SEGURO",
        bundle_confidence="high",
        is_depth_anomaly=True,
    )
    warning = ClauseTreeWarning(
        document_id="17", page_number=2, kind="depth_anomaly", detail="test warning"
    )
    report = ClauseTreeReport(
        document_id="17",
        filename="15414902071201468.pdf",
        clause_count=2,
        max_depth=2,
        orphan_char_count=3,
        total_char_count=50,
        orphan_ratio=0.06,
        extraction_mode="text",
        warnings=(warning,),
    )
    return ClauseTree(
        document_id="17",
        filename="15414902071201468.pdf",
        roots=(root,),
        all_clauses=(root, child),
        report=report,
    )


@pytest.mark.unit
def test_write_then_read_clause_tree_cache_round_trips(tmp_path: Path) -> None:
    tree = _tree()
    path = tmp_path / "17__abc123.parquet"

    write_clause_tree_cache(tree, path)
    restored = read_clause_tree_cache(path)

    assert restored == tree


@pytest.mark.unit
def test_write_then_read_clause_tree_cache_preserves_no_clauses(tmp_path: Path) -> None:
    report = ClauseTreeReport(
        document_id="9",
        filename="empty.pdf",
        clause_count=0,
        max_depth=0,
        orphan_char_count=0,
        total_char_count=0,
        orphan_ratio=0.0,
        extraction_mode="text",
        warnings=(),
    )
    tree = ClauseTree(
        document_id="9", filename="empty.pdf", roots=(), all_clauses=(), report=report
    )
    path = tmp_path / "9__abc123.parquet"

    write_clause_tree_cache(tree, path)
    restored = read_clause_tree_cache(path)

    assert restored.all_clauses == ()
    assert restored.roots == ()
    assert restored.report.clause_count == 0
