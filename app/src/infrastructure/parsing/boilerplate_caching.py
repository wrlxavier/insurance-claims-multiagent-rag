"""Parquet cache for boilerplate-removed documents.

Reuses [infrastructure.parsing.caching]'s ``write_cache``/``read_cache``,
already generic over ``ExtractedDocument``. Keyed by document id and a hash
of the removal algorithm version alone -- the upstream extraction/OCR and
normalization versions are already baked into the input artifact one hop
back (``data/cache/extraction/`` or ``data/cache/ocr/``), so they don't need
to be folded in again here.

This artifact is the contract boundary [M1-04] (clause-tree segmentation)
reads from, playing the same role for that stage that the extraction and
OCR caches play for this one.
"""

import hashlib
from pathlib import Path

BOILERPLATE_CACHE_DIR = Path("data/cache/boilerplate_removed")


def compute_boilerplate_cache_key(boilerplate_removal_version: str) -> str:
    """Hash the input that should invalidate a cached boilerplate-removed document."""
    payload = boilerplate_removal_version.encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def boilerplate_cache_path(document_id: str, cache_key: str) -> Path:
    """Return the Parquet cache path for a document/cache-key pair."""
    return BOILERPLATE_CACHE_DIR / f"{document_id}__{cache_key}.parquet"
