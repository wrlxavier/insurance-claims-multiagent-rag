"""Real-corpus regression guard for [M1-03], required by its DoD.

Extracts fresh from the real PDFs in ``data/policies/raw/`` (all three are
``extraction_mode=text`` per ``manifest.csv``, so no OCR path needed) rather
than reading the Parquet cache -- ``data/cache/extraction/`` is gitignored
and empty on a fresh clone, so a test depending on it being populated would
pass locally but fail in CI. Mirrors the fixture convention in
``test_extraction.py``.

Expected counts below are the measured numbers from the investigation that
shaped [M1-03]'s design (see the module docstring in
``application/use_cases/boilerplate_removal.py``), not arbitrary bounds.
"""

from pathlib import Path

import pytest

from application.use_cases.boilerplate_removal import remove_boilerplate
from infrastructure.parsing.extraction import PyMuPdfTextExtractor

RAW_DIR = Path(__file__).resolve().parents[4] / "data" / "policies" / "raw"


@pytest.mark.unit
def test_bradesco_207pp_has_toc_but_no_marketing_page() -> None:
    """Bradesco starts directly with the TOC on page 1 -- no cover page.
    A rule that invented one here would be over-fitting to the other two
    documents in the sample."""
    document = PyMuPdfTextExtractor().extract(RAW_DIR / "15414900666201489.pdf", "10")

    _cleaned, counts = remove_boilerplate(document)

    assert counts.header_footer_lines_removed > 0
    assert counts.toc_pages_removed == 2
    assert counts.marketing_pages_removed == 0
    assert counts.removed_page_numbers == (1, 2)


@pytest.mark.unit
def test_mapfre_146pp_detects_cover_page_and_multipage_toc() -> None:
    document = PyMuPdfTextExtractor().extract(RAW_DIR / "15414100326200483.pdf", "15")

    _cleaned, counts = remove_boilerplate(document)

    assert counts.header_footer_lines_removed > 0
    assert counts.marketing_pages_removed == 1
    assert counts.toc_pages_removed == 3
    assert counts.removed_page_numbers == (1, 2, 3, 4)


@pytest.mark.unit
def test_youse_122pp_excludes_transition_page_from_toc() -> None:
    """Page 4 is already body text (the real page where the TOC ends);
    it must not be swept in just because pages 2-3 are TOC."""
    document = PyMuPdfTextExtractor().extract(RAW_DIR / "15414900039201618.pdf", "9")

    _cleaned, counts = remove_boilerplate(document)

    assert counts.header_footer_lines_removed > 0
    assert counts.marketing_pages_removed == 1
    assert counts.toc_pages_removed == 2
    assert 4 not in counts.removed_page_numbers
    assert counts.removed_page_numbers == (1, 2, 3)
