"""Port for rasterizing PDF pages to images."""

from pathlib import Path
from typing import Protocol


class PageRasterizerPort(Protocol):
    """Interface for turning a page range of a source PDF into images."""

    def rasterize(
        self, pdf_path: Path, document_id: str, first_page: int, last_page: int
    ) -> tuple[bytes, ...]:
        """Rasterize ``pdf_path``'s pages ``[first_page, last_page]`` (1-indexed).

        Args:
            pdf_path: Path to the source PDF.
            document_id: The manifest document id, for cache keying.
            first_page: First page to rasterize, inclusive, 1-indexed.
            last_page: Last page to rasterize, inclusive, 1-indexed.

        Returns:
            One PNG-encoded image per page, in page order.
        """
        ...
