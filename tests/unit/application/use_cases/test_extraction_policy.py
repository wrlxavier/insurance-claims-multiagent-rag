import pytest

from application.use_cases.extraction_policy import (
    ExtractionRoute,
    decide_route,
    flag_low_char_pages,
)
from domain.extracted_text import ExtractedPage


@pytest.mark.unit
@pytest.mark.parametrize("extraction_mode", ["text", "partial"])
def test_decide_route_extracts_text_and_partial_now(extraction_mode: str) -> None:
    assert decide_route(extraction_mode) is ExtractionRoute.EXTRACT_NOW


@pytest.mark.unit
def test_decide_route_sends_ocr_required_to_ocr() -> None:
    assert decide_route("ocr_required") is ExtractionRoute.OCR_REQUIRED


@pytest.mark.unit
def test_decide_route_rejects_unrecognized_mode() -> None:
    with pytest.raises(ValueError, match="Unrecognized extraction_mode"):
        decide_route("scanned")


def _page(page_number: int, char_count: int) -> ExtractedPage:
    return ExtractedPage(page_number=page_number, spans=(), char_count=char_count)


@pytest.mark.unit
def test_flag_low_char_pages_flags_pages_below_threshold() -> None:
    pages = [_page(1, 500), _page(2, 10), _page(3, 39), _page(4, 40)]

    assert flag_low_char_pages(pages, threshold=40) == [2, 3]


@pytest.mark.unit
def test_flag_low_char_pages_returns_empty_when_all_pages_pass() -> None:
    pages = [_page(1, 500), _page(2, 100)]

    assert flag_low_char_pages(pages, threshold=40) == []
