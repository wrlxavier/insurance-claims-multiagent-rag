"""Real-corpus regression guard for [M1-04], required by its DoD.

Extracts fresh from the real PDFs in ``data/policies/raw/`` rather than
reading the Parquet caches (both gitignored and empty on a fresh clone),
mirroring ``test_boilerplate_removal_corpus.py``'s pattern. Text-mode
documents (14, 15, 10, 28) use the fast PyMuPDF path and are marked
``unit``; the two OCR-required documents (20, 25) shell out to the real
Tesseract binary via [infrastructure.parsing.ocr.TesseractOcrExtractor] and
are marked ``integration``, mirroring ``test_ocr.py``'s split.

Expected bounds below are the measured shape of the real trees from the
investigation that shaped [M1-04]'s heuristic (see the module docstring in
``application/use_cases/clause_segmentation.py``), not arbitrary guesses:
manifest ids 14 (Allianz, 2004) and 15 (Mapfre, 2004) are annotated in
``data/policies/manifest.csv`` as "legacy layout, structure-recovery test"
documents; id 10 is the 207-page Bradesco bundle; id 28 is the only
document confirmed to have a genuine sustained two-column layout.
"""

from functools import cache
from pathlib import Path

import pytest

from application.use_cases.boilerplate_removal import remove_boilerplate
from application.use_cases.clause_segmentation import segment_document
from domain.clause_tree import ClauseTree, HeadingConvention
from infrastructure.parsing.extraction import PyMuPdfTextExtractor
from infrastructure.parsing.ocr import TesseractOcrExtractor

RAW_DIR = Path(__file__).resolve().parents[4] / "data" / "policies" / "raw"
_OCR_DPI = 150


@cache
def _segment_text_mode(filename: str, document_id: str) -> ClauseTree:
    document = PyMuPdfTextExtractor().extract(RAW_DIR / filename, document_id)
    cleaned, _counts = remove_boilerplate(document)
    return segment_document(cleaned)


@cache
def _segment_ocr_mode(filename: str, document_id: str) -> ClauseTree:
    document = TesseractOcrExtractor(dpi=_OCR_DPI).extract(
        RAW_DIR / filename, document_id
    )
    cleaned, _counts = remove_boilerplate(document)
    return segment_document(cleaned)


@pytest.mark.unit
def test_allianz_2004_legacy_layout_recovers_deep_nested_tree() -> None:
    tree = _segment_text_mode("15414002216200457.pdf", "14")

    assert tree.report.clause_count > 0
    assert tree.report.max_depth >= 3
    assert tree.report.orphan_ratio < 0.15
    assert any(
        clause.convention == HeadingConvention.NUMBERED_DECIMAL
        for clause in tree.all_clauses
    )


@pytest.mark.unit
def test_mapfre_2004_legacy_layout_recovers_deepest_nesting() -> None:
    tree = _segment_text_mode("15414100326200483.pdf", "15")

    assert tree.report.clause_count > 0
    assert tree.report.max_depth >= 4
    assert tree.report.orphan_ratio < 0.15
    # Lettered glossary sub-items (e.g. "Terceiro" -> a)-e)) must attach as
    # content, never become their own clause: no clause title should itself
    # start with a lettered-list marker.
    assert not any(
        clause.title.startswith(("a) ", "b) ", "c) ")) for clause in tree.all_clauses
    )


@pytest.mark.unit
def test_bradesco_207pp_bundle_is_wide_and_shallow() -> None:
    tree = _segment_text_mode("15414900666201489.pdf", "10")

    assert tree.report.clause_count > 100
    # Numbering restarts (~30 embedded coverage products) must produce many
    # independent top-level siblings, not one runaway-deep subtree.
    assert len(tree.roots) >= 15
    assert 2 <= tree.report.max_depth <= 5


@pytest.mark.unit
def test_assurant_11pp_triggers_multi_column_reflow() -> None:
    tree = _segment_text_mode("15414607840202570.pdf", "28")

    assert tree.report.clause_count > 0
    assert any(
        warning.kind == "multi_column_reflow" for warning in tree.report.warnings
    )


@pytest.mark.integration
def test_kovr_ocr_document_uses_relaxed_pattern_only_detection() -> None:
    tree = _segment_ocr_mode("15414604545202481.pdf", "20")

    assert tree.report.extraction_mode == "ocr_required"
    assert any(warning.kind == "ocr_relaxed_mode" for warning in tree.report.warnings)


@pytest.mark.integration
def test_too_seguros_ocr_document_uses_relaxed_pattern_only_detection() -> None:
    tree = _segment_ocr_mode("15414618005202301.pdf", "25")

    assert tree.report.extraction_mode == "ocr_required"
    assert any(warning.kind == "ocr_relaxed_mode" for warning in tree.report.warnings)


@pytest.mark.integration
def test_ocr_documents_have_higher_orphan_ratio_than_a_comparable_text_document() -> (
    None
):
    """Expected corpus behavior, not a bug: no font/position signal on OCR
    pages means more unattached text. This is asserted, not hidden."""
    ocr_tree = _segment_ocr_mode("15414604545202481.pdf", "20")
    text_tree = _segment_text_mode("15414002216200457.pdf", "14")

    assert ocr_tree.report.orphan_ratio > text_tree.report.orphan_ratio
