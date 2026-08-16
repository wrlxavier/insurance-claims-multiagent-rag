from functools import cache
from pathlib import Path

import fitz
import pytesseract
import pytest
from PIL import Image

from domain.extracted_text import ExtractedDocument
from infrastructure.parsing.ocr import TesseractOcrExtractor

RAW_DIR = Path(__file__).resolve().parents[4] / "data" / "policies" / "raw"

# The two documents with no usable text layer, routed to OCR (M1-02).
FIXTURES = [
    pytest.param("15414604545202481.pdf", "20", 33, id="kovr"),
    pytest.param("15414618005202301.pdf", "25", 27, id="too_seguros"),
]

_DPI = 150


@cache
def _extract(filename: str, document_id: str) -> ExtractedDocument:
    """OCR a fixture once per test session; each real OCR run is expensive."""
    return TesseractOcrExtractor(dpi=_DPI).extract(RAW_DIR / filename, document_id)


# Everything below runs the real Tesseract binary against real fixture PDFs,
# so it's marked integration rather than unit -- see the "Unit coverage"
# section further down for the mocked-pytesseract coverage that runs without it.
@pytest.mark.integration
@pytest.mark.parametrize(("filename", "document_id", "expected_page_count"), FIXTURES)
def test_extract_produces_one_page_per_pdf_page(
    filename: str, document_id: str, expected_page_count: int
) -> None:
    document = _extract(filename, document_id)

    assert isinstance(document, ExtractedDocument)
    assert document.document_id == document_id
    assert document.filename == filename
    assert len(document.pages) == expected_page_count
    assert [page.page_number for page in document.pages] == list(
        range(1, expected_page_count + 1)
    )


@pytest.mark.integration
@pytest.mark.parametrize(("filename", "document_id", "_page_count"), FIXTURES)
def test_extract_produces_exactly_one_full_page_span_per_page(
    filename: str, document_id: str, _page_count: int
) -> None:
    document = _extract(filename, document_id)

    reference = fitz.open(RAW_DIR / filename)
    try:
        for page, ref_page in zip(document.pages, reference, strict=True):
            assert len(page.spans) == 1
            span = page.spans[0]
            assert span.line_id == 0
            assert span.order == 0
            assert span.document_id == document_id
            assert span.page_number == page.page_number
            assert span.font_size == 0.0
            assert span.font_name == ""
            assert span.bbox == (
                0.0,
                0.0,
                float(ref_page.rect.width),
                float(ref_page.rect.height),
            )
    finally:
        reference.close()


@pytest.mark.integration
@pytest.mark.parametrize(("filename", "document_id", "_page_count"), FIXTURES)
def test_extract_normalizes_span_text(
    filename: str, document_id: str, _page_count: int
) -> None:
    document = _extract(filename, document_id)

    for page in document.pages:
        span = page.spans[0]
        assert "\xad" not in span.text  # soft hyphen must never survive
        assert "\xa0" not in span.text  # nbsp must never survive


@pytest.mark.integration
@pytest.mark.parametrize(("filename", "document_id", "_page_count"), FIXTURES)
def test_extract_recovers_nonempty_text(
    filename: str, document_id: str, _page_count: int
) -> None:
    document = _extract(filename, document_id)

    total_chars = sum(page.char_count for page in document.pages)
    # Loose floor, not an exact digest: Tesseract output can vary slightly
    # across builds/platforms. DPI=150 stabilization was already validated
    # in notebooks/scratch/ocr_exploration.ipynb; this only checks OCR ran.
    assert total_chars > 100 * len(document.pages)


@pytest.mark.integration
@pytest.mark.parametrize(("filename", "document_id", "_page_count"), FIXTURES)
def test_extractor_version_identifies_tesseract(
    filename: str, document_id: str, _page_count: int
) -> None:
    document = _extract(filename, document_id)

    assert document.extractor_version.startswith("tesseract-")


# --- Unit coverage below: stubs out pytesseract so it runs without the
# Tesseract binary, exercising the parts of ocr.py that don't depend on real
# OCR output -- DPI/zoom math, span construction, and text normalization.
# Rasterization itself still goes through real PyMuPDF, against a synthetic
# in-memory PDF rather than the real fixtures used above.


def _make_pdf(tmp_path: Path, page_sizes: list[tuple[float, float]]) -> Path:
    """Write a minimal real PDF (blank pages, no Tesseract involved)."""
    document = fitz.open()
    for width, height in page_sizes:
        document.new_page(width=width, height=height)
    path = tmp_path / "synthetic.pdf"
    document.save(path)
    document.close()
    return path


@pytest.fixture
def _stub_tesseract_version(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub the constructor's version probe, which otherwise shells out."""
    monkeypatch.setattr(pytesseract, "get_tesseract_version", lambda: "9.9.9-fake")


@pytest.mark.unit
@pytest.mark.usefixtures("_stub_tesseract_version")
def test_extract_produces_one_page_per_pdf_page_without_tesseract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf_path = _make_pdf(tmp_path, [(100.0, 50.0), (100.0, 50.0)])
    monkeypatch.setattr(
        pytesseract, "image_to_string", lambda image, lang=None: "mocked"
    )

    document = TesseractOcrExtractor(dpi=150).extract(pdf_path, "doc-1")

    assert document.document_id == "doc-1"
    assert document.filename == pdf_path.name
    assert [page.page_number for page in document.pages] == [1, 2]


@pytest.mark.unit
@pytest.mark.usefixtures("_stub_tesseract_version")
def test_ocr_page_builds_full_page_span_with_no_position_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf_path = _make_pdf(tmp_path, [(210.0, 99.0)])
    monkeypatch.setattr(
        pytesseract, "image_to_string", lambda image, lang=None: "hello"
    )

    document = TesseractOcrExtractor(dpi=150).extract(pdf_path, "doc-1")

    span = document.pages[0].spans[0]
    assert span.document_id == "doc-1"
    assert span.page_number == 1
    assert span.line_id == 0
    assert span.order == 0
    assert span.font_size == 0.0
    assert span.font_name == ""
    assert span.bbox == (0.0, 0.0, 210.0, 99.0)
    assert span.text == "hello"
    assert document.pages[0].char_count == len("hello")


@pytest.mark.unit
@pytest.mark.usefixtures("_stub_tesseract_version")
def test_ocr_page_normalizes_raw_ocr_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf_path = _make_pdf(tmp_path, [(100.0, 50.0)])
    monkeypatch.setattr(
        pytesseract,
        "image_to_string",
        lambda image, lang=None: "so\xadft\xa0hyphen",
    )

    document = TesseractOcrExtractor(dpi=150).extract(pdf_path, "doc-1")

    text = document.pages[0].spans[0].text
    assert "\xad" not in text
    assert "\xa0" not in text


@pytest.mark.unit
@pytest.mark.usefixtures("_stub_tesseract_version")
def test_extract_rasterizes_pages_using_dpi_derived_zoom(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # dpi=144 -> zoom=2.0 exactly, so a 100x50pt page rasterizes to 200x100px.
    pdf_path = _make_pdf(tmp_path, [(100.0, 50.0)])
    captured_sizes: list[tuple[int, int]] = []

    def fake_image_to_string(image: Image.Image, lang: str | None = None) -> str:
        captured_sizes.append(image.size)
        return "text"

    monkeypatch.setattr(pytesseract, "image_to_string", fake_image_to_string)

    TesseractOcrExtractor(dpi=144).extract(pdf_path, "doc-1")

    assert captured_sizes == [(200, 100)]


@pytest.mark.unit
@pytest.mark.usefixtures("_stub_tesseract_version")
def test_extract_passes_configured_language_to_tesseract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf_path = _make_pdf(tmp_path, [(100.0, 50.0)])
    captured_langs: list[str | None] = []

    def fake_image_to_string(image: Image.Image, lang: str | None = None) -> str:
        captured_langs.append(lang)
        return "text"

    monkeypatch.setattr(pytesseract, "image_to_string", fake_image_to_string)

    TesseractOcrExtractor(dpi=150, lang="eng").extract(pdf_path, "doc-1")

    assert captured_langs == ["eng"]


@pytest.mark.unit
def test_extractor_version_uses_tesseract_binary_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pytesseract, "get_tesseract_version", lambda: "5.3.4-mocked")
    monkeypatch.setattr(pytesseract, "image_to_string", lambda image, lang=None: "text")
    pdf_path = _make_pdf(tmp_path, [(100.0, 50.0)])

    document = TesseractOcrExtractor(dpi=150).extract(pdf_path, "doc-1")

    assert document.extractor_version == "tesseract-5.3.4-mocked"
