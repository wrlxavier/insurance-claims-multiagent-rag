from pathlib import Path

import pytest

from domain.extracted_text import ExtractedDocument, ExtractedPage, ExtractedSpan
from infrastructure.parsing.caching import read_cache, write_cache
from infrastructure.parsing.ocr_caching import ocr_cache_path


@pytest.mark.unit
def test_ocr_cache_path_names_file_by_document_id_and_dpi() -> None:
    path = ocr_cache_path("20", 150)

    assert path.name == "20__dpi150.parquet"


def _ocr_document() -> ExtractedDocument:
    span = ExtractedSpan(
        document_id="20",
        page_number=1,
        line_id=0,
        order=0,
        bbox=(0.0, 0.0, 595.0, 842.0),
        font_size=0.0,
        font_name="",
        text="CONDIÇÕES GERAIS",
    )
    page = ExtractedPage(page_number=1, spans=(span,), char_count=len(span.text))
    return ExtractedDocument(
        document_id="20",
        filename="15414604545202481.pdf",
        pages=(page,),
        extractor_version="tesseract-5.3.4",
    )


@pytest.mark.unit
def test_write_then_read_ocr_cache_round_trips_full_page_span(tmp_path: Path) -> None:
    document = _ocr_document()
    path = tmp_path / "20__dpi150.parquet"

    write_cache(document, path)
    restored = read_cache(path)

    assert restored == document
