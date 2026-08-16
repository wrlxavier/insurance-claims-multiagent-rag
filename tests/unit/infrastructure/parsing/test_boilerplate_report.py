import pytest

from infrastructure.parsing.boilerplate_report import (
    BoilerplateReportRow,
    render_report,
)


@pytest.mark.unit
def test_render_report_includes_extracted_document_stats() -> None:
    row = BoilerplateReportRow(
        document_id="10",
        filename="15414900666201489.pdf",
        page_count=207,
        header_footer_lines_removed=621,
        toc_dot_leader_lines_removed=0,
        toc_pages_removed=2,
        toc_page_lines_removed=124,
        marketing_pages_removed=0,
        marketing_page_lines_removed=0,
        total_chars_before=802167,
        total_chars_after=772543,
        removed_page_numbers=(1, 2),
    )

    report = render_report([row])

    assert "15414900666201489.pdf" in report
    assert "207" in report
    assert "621" in report
    assert "802167" in report
    assert "772543" in report
    assert "1, 2" in report


@pytest.mark.unit
def test_render_report_shows_none_for_documents_with_no_removed_pages() -> None:
    row = BoilerplateReportRow(
        document_id="17",
        filename="a.pdf",
        page_count=32,
        header_footer_lines_removed=128,
        toc_dot_leader_lines_removed=0,
        toc_pages_removed=0,
        toc_page_lines_removed=0,
        marketing_pages_removed=0,
        marketing_page_lines_removed=0,
        total_chars_before=120673,
        total_chars_after=116842,
        removed_page_numbers=(),
    )

    report = render_report([row])

    assert "| none |" in report


@pytest.mark.unit
def test_render_report_summarizes_counts_across_documents() -> None:
    row_a = BoilerplateReportRow(
        document_id="10",
        filename="a.pdf",
        page_count=207,
        header_footer_lines_removed=621,
        toc_dot_leader_lines_removed=0,
        toc_pages_removed=2,
        toc_page_lines_removed=124,
        marketing_pages_removed=0,
        marketing_page_lines_removed=0,
        total_chars_before=802167,
        total_chars_after=772543,
        removed_page_numbers=(1, 2),
    )
    row_b = BoilerplateReportRow(
        document_id="15",
        filename="b.pdf",
        page_count=146,
        header_footer_lines_removed=874,
        toc_dot_leader_lines_removed=0,
        toc_pages_removed=3,
        toc_page_lines_removed=162,
        marketing_pages_removed=1,
        marketing_page_lines_removed=41,
        total_chars_before=387414,
        total_chars_after=343971,
        removed_page_numbers=(1, 2, 3, 4),
    )

    report = render_report([row_a, row_b])

    assert "Header/footer lines removed: 1495" in report
    assert "TOC pages removed: 5" in report
    assert "Marketing/cover pages removed: 1 (1/2 documents)" in report
    assert "Total characters: 1189581 → 1116514" in report
    assert "Documents out of 2 total." in report
