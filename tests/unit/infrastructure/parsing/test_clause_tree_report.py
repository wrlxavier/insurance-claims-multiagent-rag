import pytest

from infrastructure.parsing.clause_tree_report import ClauseTreeReportRow, render_report


@pytest.mark.unit
def test_render_report_includes_document_stats() -> None:
    row = ClauseTreeReportRow(
        document_id="10",
        filename="15414900666201489.pdf",
        page_count=207,
        clause_count=214,
        max_depth=3,
        orphan_ratio=0.021,
        extraction_mode="text",
        warning_count=0,
        exceeds_threshold=False,
    )

    report = render_report([row], threshold=0.15)

    assert "15414900666201489.pdf" in report
    assert "207" in report
    assert "214" in report
    assert "0.021" in report
    assert "text" in report


@pytest.mark.unit
def test_render_report_flags_documents_exceeding_threshold() -> None:
    row = ClauseTreeReportRow(
        document_id="20",
        filename="kovr.pdf",
        page_count=33,
        clause_count=4,
        max_depth=1,
        orphan_ratio=0.42,
        extraction_mode="ocr_required",
        warning_count=2,
        exceeds_threshold=True,
    )

    report = render_report([row], threshold=0.15)

    assert "0.420 ⚠" in report
    assert "Documents exceeding the orphan-ratio threshold: 1 (20)" in report


@pytest.mark.unit
def test_render_report_summarizes_counts_across_documents() -> None:
    row_a = ClauseTreeReportRow(
        document_id="14",
        filename="a.pdf",
        page_count=116,
        clause_count=87,
        max_depth=4,
        orphan_ratio=0.02,
        extraction_mode="text",
        warning_count=0,
        exceeds_threshold=False,
    )
    row_b = ClauseTreeReportRow(
        document_id="15",
        filename="b.pdf",
        page_count=146,
        clause_count=133,
        max_depth=4,
        orphan_ratio=0.03,
        extraction_mode="text",
        warning_count=1,
        exceeds_threshold=False,
    )

    report = render_report([row_a, row_b], threshold=0.15)

    assert "Total clauses recovered: 220" in report
    assert "Documents exceeding the orphan-ratio threshold: 0" in report
    assert "Documents out of 2 total." in report
