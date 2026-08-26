"""Tests for the Markdown chunking report."""

from dataclasses import replace

import pytest

from infrastructure.rag.chunk_report import ChunkReportRow, render_chunk_report


def _row() -> ChunkReportRow:
    return ChunkReportRow(
        document_id="1",
        filename="policy.pdf",
        clause_count=10,
        chunk_count=8,
        single_count=5,
        merged_count=2,
        item_boundary_split_count=1,
        sliding_window_split_count=0,
        min_char_count=40,
        p50_char_count=300,
        p90_char_count=900,
        max_char_count=1200,
    )


@pytest.mark.unit
def test_render_chunk_report_includes_row_and_summary_totals() -> None:
    second = replace(
        _row(),
        document_id="2",
        chunk_count=4,
        single_count=3,
        merged_count=1,
        item_boundary_split_count=0,
    )
    rows = [_row(), second]

    report = render_chunk_report(rows)

    assert "| 1 | policy.pdf | 10 | 8 | 5 | 2 | 1 | 0 |" in report
    assert "## Summary" in report
    assert "Total chunks: 12" in report
    assert "Single (default, one clause -> one chunk): 8" in report
    assert "Merged (short clause(s) folded into a parent): 3" in report
    assert "Item-boundary split: 1" in report
    assert "Sliding-window split (last resort): 0" in report
    assert "Documents: 2." in report
