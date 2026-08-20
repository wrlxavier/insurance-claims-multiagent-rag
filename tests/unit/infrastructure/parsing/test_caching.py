from pathlib import Path

import pytest

from domain.extracted_text import ExtractedDocument, ExtractedPage, ExtractedSpan
from infrastructure.parsing.caching import (
    cache_path,
    compute_cache_key,
    read_cache,
    write_cache,
)


@pytest.mark.unit
def test_compute_cache_key_is_deterministic() -> None:
    assert compute_cache_key("1.28.0", "v1") == compute_cache_key("1.28.0", "v1")


@pytest.mark.unit
def test_compute_cache_key_changes_with_extractor_version() -> None:
    assert compute_cache_key("1.28.0", "v1") != compute_cache_key("1.29.0", "v1")


@pytest.mark.unit
def test_compute_cache_key_changes_with_normalization_version() -> None:
    assert compute_cache_key("1.28.0", "v1") != compute_cache_key("1.28.0", "v2")


@pytest.mark.unit
def test_cache_path_names_file_by_document_id_and_key() -> None:
    path = cache_path("17", "abc123")

    assert path.name == "17__abc123.parquet"


def _document() -> ExtractedDocument:
    span_a = ExtractedSpan(
        document_id="17",
        page_number=1,
        line_id=0,
        order=0,
        bbox=(10.0, 20.0, 100.0, 30.0),
        font_size=12.0,
        font_name="ArialMT",
        text="Cláusula Primeira",
    )
    span_b = ExtractedSpan(
        document_id="17",
        page_number=1,
        line_id=1,
        order=1,
        bbox=(10.0, 40.0, 100.0, 50.0),
        font_size=10.0,
        font_name="Arial",
        text="Do objeto do seguro.",
    )
    spans = (span_a, span_b)
    page_one = ExtractedPage(
        page_number=1, spans=spans, char_count=sum(len(s.text) for s in spans)
    )
    page_two = ExtractedPage(page_number=2, spans=(), char_count=0)
    return ExtractedDocument(
        document_id="17",
        filename="15414902071201468.pdf",
        pages=(page_one, page_two),
        extractor_version="1.28.0",
    )


@pytest.mark.unit
def test_write_then_read_cache_round_trips_the_document(tmp_path: Path) -> None:
    document = _document()
    path = tmp_path / "17__abc123.parquet"

    write_cache(document, path)
    restored = read_cache(path)

    assert restored == document


@pytest.mark.unit
def test_write_then_read_cache_preserves_empty_pages(tmp_path: Path) -> None:
    document = _document()
    path = tmp_path / "17__abc123.parquet"

    write_cache(document, path)
    restored = read_cache(path)

    assert len(restored.pages) == 2
    assert restored.pages[1].page_number == 2
    assert restored.pages[1].spans == ()
    assert restored.pages[1].char_count == 0
