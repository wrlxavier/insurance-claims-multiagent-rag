#!/usr/bin/env python3
"""Extract normalized, position-aware text from every policy PDF.

Dispatches each document in ``data/policies/manifest.csv`` by its
``extraction_mode`` column: ``text``/``partial`` documents are extracted now
via PyMuPDF and cached as Parquet; ``ocr_required`` documents are routed to
the not-yet-built [M1-02] OCR path and skipped here. Extraction output is
cached under ``data/cache/extraction/``, keyed by document id and a hash of
(PyMuPDF version, normalization version), so unchanged inputs are not
re-extracted. Writes a per-document character report to
``docs/TEXT_EXTRACTION_REPORT.md``.
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


def run_extraction(low_char_page_threshold: int) -> list[ExtractionReportRow]:
    """Dispatch every manifest row and return its report row."""
    rows: list[ExtractionReportRow] = []
    for entry in read_manifest(MANIFEST_PATH):
        document_id = entry["id"]
        filename = entry["filename"]
        route = decide_route(entry["extraction_mode"])

        if route is ExtractionRoute.OCR_REQUIRED:
            print(f"{filename}: routed to OCR, not extracted (see M1-02)")
            rows.append(
                ExtractionReportRow(
                    document_id=document_id, filename=filename, route=route
                )
            )
            continue

        document, was_cached = extract_or_load_from_cache(
            document_id, RAW_DIR / filename
        )
        page_count = len(document.pages)
        total_chars = sum(page.char_count for page in document.pages)
        flagged = flag_low_char_pages(document.pages, low_char_page_threshold)
        cache_note = "cached" if was_cached else "extracted"
        print(
            f"{filename}: {cache_note}, {page_count} pages, {total_chars} chars, "
            f"{len(flagged)} flagged"
        )
        rows.append(
            ExtractionReportRow(
                document_id=document_id,
                filename=filename,
                route=route,
                page_count=page_count,
                total_chars=total_chars,
                avg_chars_per_page=total_chars / page_count if page_count else 0.0,
                flagged_pages=tuple(flagged),
            )
        )
    return rows


def main() -> None:
    """Run extraction dispatch over the corpus and write the report."""
    threshold = get_parsing_settings().low_char_page_threshold
    rows = run_extraction(threshold)
    REPORT_PATH.write_text(render_report(rows, threshold), encoding="utf-8")


if __name__ == "__main__":
    main()
