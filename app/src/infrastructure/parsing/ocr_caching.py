"""Parquet cache for OCR output, keyed by document id and DPI.

Separate from [infrastructure.parsing.caching]'s extractor-version-hash
scheme: OCR output is governed by DPI (the tunable knob for this pipeline),
not by an extractor/normalization version hash, so the cache key is just
``(document_id, dpi)``. Reuses that module's ``write_cache``/``read_cache``,
which are already generic over ``ExtractedDocument`` and need no change.

Because the key has no version hash, upgrading Tesseract or changing
``normalize_text`` does not invalidate this cache automatically -- run
``rm -rf data/cache/ocr/`` manually after such a change.
"""

from pathlib import Path

OCR_CACHE_DIR = Path("data/cache/ocr")


def ocr_cache_path(document_id: str, dpi: int) -> Path:
    """Return the Parquet cache path for a document/DPI pair."""
    return OCR_CACHE_DIR / f"{document_id}__dpi{dpi}.parquet"
