#!/usr/bin/env python3
"""Extract normalized, position-aware text from every policy PDF.

Dispatches each document in ``data/policies/manifest.csv`` by its
``extraction_mode`` column: ``text``/``partial`` documents are extracted via
PyMuPDF and cached as Parquet under ``data/cache/extraction/``, keyed by
document id and a hash of (PyMuPDF version, normalization version);
``ocr_required`` documents (no usable text layer) are rasterized and OCR'd
via Tesseract [M1-02] and cached separately under ``data/cache/ocr/``, keyed
by document id and DPI. Either way, unchanged inputs are not re-processed.
Writes a per-document character report to ``docs/TEXT_EXTRACTION_REPORT.md``.
"""

from pathlib import Path

from application.use_cases.extraction_policy import (
    ExtractionRoute,
    decide_route,
    flag_low_char_pages,
)
from domain.extracted_text import ExtractedDocument
from infrastructure.config.settings import get_parsing_settings
from infrastructure.parsing.caching import (
    cache_path,
    compute_cache_key,
    read_cache,
    write_cache,
)
from infrastructure.parsing.extraction import PYMUPDF_VERSION, PyMuPdfTextExtractor
from infrastructure.parsing.manifest import read_manifest
from infrastructure.parsing.normalization import NORMALIZATION_VERSION
from infrastructure.parsing.ocr import TesseractOcrExtractor
from infrastructure.parsing.ocr_caching import ocr_cache_path
from infrastructure.parsing.report import ExtractionReportRow, render_report

RAW_DIR = Path("data/policies/raw")
MANIFEST_PATH = Path("data/policies/manifest.csv")
REPORT_PATH = Path("docs/TEXT_EXTRACTION_REPORT.md")

_extractor = PyMuPdfTextExtractor()


def extract_or_load_from_cache(
    document_id: str, pdf_path: Path
) -> tuple[ExtractedDocument, bool]:
    """Return the extracted document and whether it came from the cache."""
    cache_key = compute_cache_key(PYMUPDF_VERSION, NORMALIZATION_VERSION)
    path = cache_path(document_id, cache_key)
    if path.exists():
        return read_cache(path), True
    document = _extractor.extract(pdf_path, document_id)
    write_cache(document, path)
    return document, False


def ocr_or_load_from_cache(
    document_id: str, pdf_path: Path, dpi: int
) -> tuple[ExtractedDocument, bool]:
    """Return the OCR'd document and whether it came from the cache."""
    path = ocr_cache_path(document_id, dpi)
    if path.exists():
        return read_cache(path), True
    document = TesseractOcrExtractor(dpi=dpi).extract(pdf_path, document_id)
    write_cache(document, path)
    return document, False


def _build_report_row(
    document_id: str,
    filename: str,
    route: ExtractionRoute,
    document: ExtractedDocument,
    was_cached: bool,
    low_char_page_threshold: int,
) -> ExtractionReportRow:
    """Compute stats for one processed document and print a progress line."""
    page_count = len(document.pages)
    total_chars = sum(page.char_count for page in document.pages)
    flagged = flag_low_char_pages(document.pages, low_char_page_threshold)
    cache_note = "cached" if was_cached else "extracted"
    print(
        f"{filename}: {cache_note} ({route.name}), {page_count} pages, "
        f"{total_chars} chars, {len(flagged)} flagged"
    )
    return ExtractionReportRow(
        document_id=document_id,
        filename=filename,
        route=route,
        page_count=page_count,
        total_chars=total_chars,
        avg_chars_per_page=total_chars / page_count if page_count else 0.0,
        flagged_pages=tuple(flagged),
    )


def run_extraction(
    low_char_page_threshold: int, ocr_dpi: int
) -> list[ExtractionReportRow]:
    """Dispatch every manifest row and return its report row."""
    rows: list[ExtractionReportRow] = []
    for entry in read_manifest(MANIFEST_PATH):
        document_id = entry["id"]
        filename = entry["filename"]
        route = decide_route(entry["extraction_mode"])
        pdf_path = RAW_DIR / filename

        if route is ExtractionRoute.OCR_REQUIRED:
            document, was_cached = ocr_or_load_from_cache(
                document_id, pdf_path, ocr_dpi
            )
        else:
            document, was_cached = extract_or_load_from_cache(document_id, pdf_path)

        rows.append(
            _build_report_row(
                document_id,
                filename,
                route,
                document,
                was_cached,
                low_char_page_threshold,
            )
        )
    return rows


def main() -> None:
    """Run extraction dispatch over the corpus and write the report."""
    settings = get_parsing_settings()
    rows = run_extraction(settings.low_char_page_threshold, settings.ocr_dpi)
    REPORT_PATH.write_text(
        render_report(rows, settings.low_char_page_threshold), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
