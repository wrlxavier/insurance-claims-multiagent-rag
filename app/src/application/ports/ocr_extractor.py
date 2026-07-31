"""Port for extracting text from a document with no usable text layer.

This is the interface boundary for the OCR path built in [M1-02]. No
implementation exists yet: documents routed here by ``decide_route`` are
skipped, not extracted, until an adapter implementing this port lands.
"""

from pathlib import Path
from typing import Protocol

from domain.extracted_text import ExtractedDocument


class OcrExtractor(Protocol):
    """Extracts a position-aware text artifact via OCR."""

    def extract(self, pdf_path: Path, document_id: str) -> ExtractedDocument:
        """Extract normalized, position-aware text by OCR-ing ``pdf_path``."""
        ...
