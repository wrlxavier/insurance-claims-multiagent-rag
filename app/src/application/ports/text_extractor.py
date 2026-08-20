"""Port for extracting text from a document with a usable text layer."""

from pathlib import Path
from typing import Protocol

from domain.extracted_text import ExtractedDocument


class TextExtractor(Protocol):
    """Extracts a position-aware text artifact from a PDF's text layer."""

    def extract(self, pdf_path: Path, document_id: str) -> ExtractedDocument:
        """Extract normalized, position-aware text from ``pdf_path``."""
        ...
