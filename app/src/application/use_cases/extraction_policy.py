"""Pure decision logic for the text-extraction pipeline.

Kept free of any extractor or I/O dependency so both functions are testable
with plain values.
"""

from collections.abc import Sequence
from enum import Enum, auto

from domain.extracted_text import ExtractedPage


class ExtractionRoute(Enum):
    """Which path a document should take, based on its extraction_mode."""

    EXTRACT_NOW = auto()
    OCR_REQUIRED = auto()


def decide_route(extraction_mode: str) -> ExtractionRoute:
    """Route a document by its manifest ``extraction_mode`` column.

    ``text`` and ``partial`` both mean the PyMuPDF text layer is usable now.
    ``ocr_required`` means there is no usable text layer and the document
    must go through the [M1-02] OCR path instead.
    """
    if extraction_mode in ("text", "partial"):
        return ExtractionRoute.EXTRACT_NOW
    if extraction_mode == "ocr_required":
        return ExtractionRoute.OCR_REQUIRED
    raise ValueError(f"Unrecognized extraction_mode: {extraction_mode!r}")


def flag_low_char_pages(pages: Sequence[ExtractedPage], threshold: int) -> list[int]:
    """Return page numbers whose extracted character count is below threshold.

    Applies even to documents whose manifest verdict is ``text``, since a
    single low page can indicate a tooling regression the document-level
    audit would not catch.
    """
    return [page.page_number for page in pages if page.char_count < threshold]
