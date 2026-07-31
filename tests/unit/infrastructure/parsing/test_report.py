import pytest

from application.use_cases.extraction_policy import ExtractionRoute
from infrastructure.parsing.report import ExtractionReportRow, render_report


@pytest.mark.unit
def test_render_report_includes_extracted_document_stats() -> None:
    row = ExtractionReportRow(
        document_id="17",
        filename="15414902071201468.pdf",
        route=ExtractionRoute.EXTRACT_NOW,
        page_count=32,
        total_chars=64000,
        avg_chars_per_page=2000.0,
        flagged_pages=(5, 12),
    )

    report = render_report([row], threshold=40)

    assert "15414902071201468.pdf" in report
    assert "32" in report
    assert "64000" in report
    assert "2000.0" in report
    assert "5, 12" in report


@pytest.mark.unit
def test_render_report_marks_ocr_required_rows_as_not_extracted() -> None:
    row = ExtractionReportRow(
        document_id="20",
        filename="15414604545202481.pdf",
        route=ExtractionRoute.OCR_REQUIRED,
    )

    report = render_report([row], threshold=40)

    assert "routed to OCR, not extracted (see M1-02)" in report


@pytest.mark.unit
def test_render_report_summarizes_routes_and_flags() -> None:
    extracted = ExtractionReportRow(
        document_id="17",
        filename="a.pdf",
        route=ExtractionRoute.EXTRACT_NOW,
        page_count=10,
        total_chars=1000,
        avg_chars_per_page=100.0,
        flagged_pages=(3,),
    )
    ocr_required = ExtractionReportRow(
        document_id="20",
        filename="b.pdf",
        route=ExtractionRoute.OCR_REQUIRED,
    )

    report = render_report([extracted, ocr_required], threshold=40)

    assert "Extracted now: 1" in report
    assert "Routed to OCR (not extracted, see M1-02): 1" in report
    assert "Documents out of 2 total." in report
    assert "Flagged low-character pages across all extracted documents: 1" in report
