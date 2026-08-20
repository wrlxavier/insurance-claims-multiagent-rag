#!/usr/bin/env python3
"""Strip boilerplate from every already-extracted policy document.

Post-processing stage for [M1-03], running after [M1-01]/[M1-02] and before
clause-tree segmentation [M1-04]. Reads each document from the extraction
cache (``data/cache/extraction/``) or the OCR cache (``data/cache/ocr/``),
depending on its manifest ``extraction_mode`` -- this script never invokes
PyMuPDF or Tesseract itself, since a missing upstream cache means
``scripts/extract_text.py`` has not been run yet, and re-extracting silently
here would hide that instead of failing loudly. Removes repeated
headers/footers, table-of-contents lines and front-matter marketing/cover
pages, caches the result under ``data/cache/boilerplate_removed/``, and
writes a per-document removal report to
``docs/BOILERPLATE_REMOVAL_REPORT.md``.
"""

from pathlib import Path

from application.use_cases.boilerplate_removal import (
    BOILERPLATE_REMOVAL_VERSION,
    BoilerplateRemovalCounts,
    remove_boilerplate,
)
from application.use_cases.extraction_policy import ExtractionRoute, decide_route
from domain.extracted_text import ExtractedDocument
from infrastructure.config.settings import get_parsing_settings
from infrastructure.parsing.boilerplate_caching import (
    boilerplate_cache_path,
    compute_boilerplate_cache_key,
)
from infrastructure.parsing.boilerplate_report import (
    BoilerplateReportRow,
    render_report,
)
from infrastructure.parsing.caching import (
    cache_path,
    compute_cache_key,
    read_cache,
)
from infrastructure.parsing.caching import write_cache as write_cleaned_cache
from infrastructure.parsing.extraction import PYMUPDF_VERSION
from infrastructure.parsing.manifest import read_manifest
from infrastructure.parsing.normalization import NORMALIZATION_VERSION
from infrastructure.parsing.ocr_caching import ocr_cache_path

MANIFEST_PATH = Path("data/policies/manifest.csv")
REPORT_PATH = Path("docs/BOILERPLATE_REMOVAL_REPORT.md")


def load_extracted_document(
    document_id: str, filename: str, extraction_mode: str, ocr_dpi: int
) -> ExtractedDocument:
    """Load a document from its upstream [M1-01]/[M1-02] cache.

    Raises ``FileNotFoundError`` if that cache is missing -- run
    ``scripts/extract_text.py`` first.
    """
    route = decide_route(extraction_mode)
    if route is ExtractionRoute.OCR_REQUIRED:
        path = ocr_cache_path(document_id, ocr_dpi)
    else:
        cache_key = compute_cache_key(PYMUPDF_VERSION, NORMALIZATION_VERSION)
        path = cache_path(document_id, cache_key)

    if not path.exists():
        raise FileNotFoundError(
            f"No extraction cache for {filename} (document {document_id}) at "
            f"{path}. Run `scripts/extract_text.py` first."
        )
    return read_cache(path)


def remove_boilerplate_and_cache(
    document: ExtractedDocument,
) -> tuple[ExtractedDocument, BoilerplateRemovalCounts, bool]:
    """Run removal and cache the result. Returns (doc, counts, was_cached).

    Removal always runs in memory, even on a cache hit, so the report's
    per-document counts stay accurate every run -- only the Parquet
    *write* is skipped when the cache file already exists. This is a
    deliberate deviation from the extraction/OCR caches, which are gated
    on cache hits because their underlying computation (PyMuPDF/Tesseract)
    is expensive; removal is a cheap in-memory pass.
    """
    cleaned_document, counts = remove_boilerplate(document)
    cache_key = compute_boilerplate_cache_key(BOILERPLATE_REMOVAL_VERSION)
    path = boilerplate_cache_path(document.document_id, cache_key)
    was_cached = path.exists()
    if not was_cached:
        write_cleaned_cache(cleaned_document, path)
    return cleaned_document, counts, was_cached


def _build_report_row(
    document_id: str,
    filename: str,
    original_document: ExtractedDocument,
    cleaned_document: ExtractedDocument,
    counts: BoilerplateRemovalCounts,
    was_cached: bool,
) -> BoilerplateReportRow:
    """Compute stats for one processed document and print a progress line."""
    chars_before = sum(page.char_count for page in original_document.pages)
    chars_after = sum(page.char_count for page in cleaned_document.pages)
    cache_note = "cached" if was_cached else "written"
    print(
        f"{filename}: {cache_note}, {len(original_document.pages)} pages, "
        f"removed: header/footer={counts.header_footer_lines_removed} "
        f"toc_dot_leader={counts.toc_dot_leader_lines_removed} "
        f"toc_pages={counts.toc_pages_removed} "
        f"marketing_pages={counts.marketing_pages_removed}"
    )
    return BoilerplateReportRow(
        document_id=document_id,
        filename=filename,
        page_count=len(original_document.pages),
        header_footer_lines_removed=counts.header_footer_lines_removed,
        toc_dot_leader_lines_removed=counts.toc_dot_leader_lines_removed,
        toc_pages_removed=counts.toc_pages_removed,
        toc_page_lines_removed=counts.toc_page_lines_removed,
        marketing_pages_removed=counts.marketing_pages_removed,
        marketing_page_lines_removed=counts.marketing_page_lines_removed,
        total_chars_before=chars_before,
        total_chars_after=chars_after,
        removed_page_numbers=counts.removed_page_numbers,
    )


def run_boilerplate_removal(ocr_dpi: int) -> list[BoilerplateReportRow]:
    """Process every manifest row and return its report row."""
    rows: list[BoilerplateReportRow] = []
    for entry in read_manifest(MANIFEST_PATH):
        document_id = entry["id"]
        filename = entry["filename"]
        original_document = load_extracted_document(
            document_id, filename, entry["extraction_mode"], ocr_dpi
        )
        cleaned_document, counts, was_cached = remove_boilerplate_and_cache(
            original_document
        )
        rows.append(
            _build_report_row(
                document_id,
                filename,
                original_document,
                cleaned_document,
                counts,
                was_cached,
            )
        )
    return rows


def main() -> None:
    """Run boilerplate removal over the corpus and write the report."""
    settings = get_parsing_settings()
    rows = run_boilerplate_removal(settings.ocr_dpi)
    REPORT_PATH.write_text(render_report(rows), encoding="utf-8")


if __name__ == "__main__":
    main()
