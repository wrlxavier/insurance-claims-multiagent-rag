from pathlib import Path

import pytest

from domain.extracted_text import ExtractedDocument
from infrastructure.parsing.extraction import PyMuPdfTextExtractor

RAW_DIR = Path(__file__).resolve().parents[4] / "data" / "policies" / "raw"

# One fixture per layout era present in the corpus, all extraction_mode=text.
FIXTURES = [
    pytest.param("15414000861200605.pdf", "30", 9, id="2004-2006_era"),
    pytest.param("15414902071201468.pdf", "17", 32, id="mid-2010s_era"),
    pytest.param("15414634764202494.pdf", "21", 28, id="2023-2024_era"),
]


@pytest.mark.unit
@pytest.mark.parametrize(("filename", "document_id", "expected_page_count"), FIXTURES)
def test_extract_produces_one_page_per_pdf_page(
    filename: str, document_id: str, expected_page_count: int
) -> None:
    document = PyMuPdfTextExtractor().extract(RAW_DIR / filename, document_id)

    assert isinstance(document, ExtractedDocument)
    assert document.document_id == document_id
    assert document.filename == filename
    assert len(document.pages) == expected_page_count
    assert [page.page_number for page in document.pages] == list(
        range(1, expected_page_count + 1)
    )


@pytest.mark.unit
@pytest.mark.parametrize(("filename", "document_id", "_page_count"), FIXTURES)
def test_extract_preserves_span_level_position_and_font_metadata(
    filename: str, document_id: str, _page_count: int
) -> None:
    document = PyMuPdfTextExtractor().extract(RAW_DIR / filename, document_id)

    all_spans = [span for page in document.pages for span in page.spans]
    assert all_spans, "expected at least one span"

    for span in all_spans:
        assert span.document_id == document_id
        assert 1 <= span.page_number <= len(document.pages)
        assert span.font_size > 0
        assert span.font_name
        x0, y0, x1, y1 = span.bbox
        assert x1 >= x0
        assert y1 >= y0

    # Span-level (not block-level) fidelity: font size varies across spans
    # within this document, so a per-block aggregate would have lost it.
    distinct_font_sizes = {round(span.font_size, 1) for span in all_spans}
    assert len(distinct_font_sizes) > 1


@pytest.mark.unit
@pytest.mark.parametrize(("filename", "document_id", "_page_count"), FIXTURES)
def test_extract_assigns_monotonic_order_and_line_id_per_page(
    filename: str, document_id: str, _page_count: int
) -> None:
    document = PyMuPdfTextExtractor().extract(RAW_DIR / filename, document_id)

    for page in document.pages:
        orders = [span.order for span in page.spans]
        assert orders == sorted(orders)
        assert orders == list(range(len(page.spans)))
        assert all(span.line_id >= 0 for span in page.spans)


@pytest.mark.unit
@pytest.mark.parametrize(("filename", "document_id", "_page_count"), FIXTURES)
def test_extract_normalizes_span_text(
    filename: str, document_id: str, _page_count: int
) -> None:
    document = PyMuPdfTextExtractor().extract(RAW_DIR / filename, document_id)

    for page in document.pages:
        for span in page.spans:
            assert "\xad" not in span.text  # soft hyphen must never survive
            assert "\xa0" not in span.text  # nbsp must never survive
