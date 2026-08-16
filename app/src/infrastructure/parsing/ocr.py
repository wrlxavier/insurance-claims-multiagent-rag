"""Tesseract-based OCR extraction, isolated behind the OcrExtractor port.

Used for documents with no usable text layer (subset fonts with no
ToUnicode map) -- see [M1-02]. Nothing outside this module (and its sibling
parsing modules) imports ``fitz``, ``pytesseract`` or ``PIL`` directly;
``domain`` and ``application`` only see the port and the entities in
[domain.extracted_text].

Each OCR'd page becomes a single full-page ``ExtractedSpan``: OCR via
``pytesseract.image_to_string`` gives no per-word position, so ``bbox`` is
the page's own rect (an honest "no real position data" placeholder, not a
provenance marker) and ``font_size``/``font_name`` are left empty (OCR has
no font metadata). Positional fidelity for OCR'd pages is deferred to a
future milestone if it is ever needed.
"""

import io
from pathlib import Path

import fitz
import pytesseract
from PIL import Image

from application.ports.ocr_extractor import OcrExtractor
from domain.extracted_text import ExtractedDocument, ExtractedPage, ExtractedSpan
from infrastructure.parsing.normalization import normalize_text

DEFAULT_OCR_LANG = "por"


class TesseractOcrExtractor(OcrExtractor):
    """Rasterizes each page (PyMuPDF) and OCRs it (Tesseract via pytesseract)."""

    def __init__(self, dpi: int, lang: str = DEFAULT_OCR_LANG) -> None:
        """Configure rasterization DPI and Tesseract language."""
        self._dpi = dpi
        self._lang = lang
        # Computed lazily (not as a module-level constant like PYMUPDF_VERSION)
        # because it shells out to the `tesseract` binary -- a module-level
        # constant would make importing this module fail without Tesseract
        # installed, even for callers that never construct an extractor.
        self._extractor_version = f"tesseract-{pytesseract.get_tesseract_version()}"

    def extract(self, pdf_path: Path, document_id: str) -> ExtractedDocument:
        """Rasterize and OCR every page of ``pdf_path`` at ``self._dpi``."""
        document = fitz.open(pdf_path)
        try:
            pages = tuple(self._ocr_page(page, document_id) for page in document)
        finally:
            document.close()
        return ExtractedDocument(
            document_id=document_id,
            filename=pdf_path.name,
            pages=pages,
            extractor_version=self._extractor_version,
        )

    def _ocr_page(self, page: fitz.Page, document_id: str) -> ExtractedPage:
        """OCR one page into a single full-page span."""
        zoom = self._dpi / 72
        pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        with Image.open(io.BytesIO(pixmap.tobytes("png"))) as image:
            raw_text = pytesseract.image_to_string(image, lang=self._lang)
        text = normalize_text(raw_text)
        page_number = page.number + 1
        span = ExtractedSpan(
            document_id=document_id,
            page_number=page_number,
            line_id=0,
            order=0,
            bbox=(0.0, 0.0, float(page.rect.width), float(page.rect.height)),
            font_size=0.0,
            font_name="",
            text=text,
        )
        return ExtractedPage(
            page_number=page_number, spans=(span,), char_count=len(text)
        )
