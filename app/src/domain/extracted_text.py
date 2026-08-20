"""Position-aware text extracted from a policy document.

Shared contract between the PyMuPDF text-extraction path and the future
OCR path (M1-02): both must produce an ``ExtractedDocument`` so that
downstream heading and clause-tree recovery (M1-04) can consume either
without caring which extractor produced it. Frozen dataclasses only, no
third-party imports, so this stays importable with nothing but the
standard library.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ExtractedSpan:
    """A single run of same-formatting text, positioned on a page."""

    document_id: str
    page_number: int
    line_id: int
    order: int
    bbox: tuple[float, float, float, float]
    font_size: float
    font_name: str
    text: str


@dataclass(frozen=True)
class ExtractedPage:
    """All spans extracted from one page, in reading order."""

    page_number: int
    spans: tuple[ExtractedSpan, ...]
    char_count: int


@dataclass(frozen=True)
class ExtractedDocument:
    """The full position-aware text artifact for one document."""

    document_id: str
    filename: str
    pages: tuple[ExtractedPage, ...]
    extractor_version: str
