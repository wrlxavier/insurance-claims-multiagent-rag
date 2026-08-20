"""Content-addressed, on-disk cache wrapping a BoundaryVisionReviewerPort.

Mirrors [infrastructure.parsing.llm_classification_cache.
CachingClauseClassifier]'s pattern exactly: a re-run against unchanged
input must not silently reshuffle a clause's boundary_source or re-pay for
a vision call already made, preserving [M1-07]'s clause_id determinism
guarantee. Keyed by a hash of (model, clause title, claimed page range, page
image bytes) -- the DoD's exact "hash of (model, page images,
deterministic-pass claim)" requirement.
"""

import hashlib
import json
import threading
from pathlib import Path

from application.ports.boundary_vision_reviewer import BoundaryVisionReviewerPort
from domain.boundary_escalation import BoundaryReview


def _cache_key(
    model: str,
    clause_title: str,
    claimed_page_start: int,
    claimed_page_end: int,
    page_images: tuple[bytes, ...],
) -> str:
    image_hashes = "|".join(hashlib.sha256(img).hexdigest() for img in page_images)
    payload = (
        f"{model}\x00{clause_title}\x00{claimed_page_start}\x00"
        f"{claimed_page_end}\x00{image_hashes}"
    ).encode()
    return hashlib.sha256(payload).hexdigest()[:32]


class CachingBoundaryVisionReviewer:
    """A [BoundaryVisionReviewerPort] that persists results to a JSON Lines file."""

    def __init__(
        self, inner: BoundaryVisionReviewerPort, model: str, cache_path: Path
    ) -> None:
        """Wrap ``inner``, caching its results under ``cache_path``."""
        self._inner = inner
        self._model = model
        self._cache_path = cache_path
        self._lock = threading.Lock()
        self._cache: dict[str, BoundaryReview] = self._load()

    def _load(self) -> dict[str, BoundaryReview]:
        if not self._cache_path.exists():
            return {}
        cache: dict[str, BoundaryReview] = {}
        with self._cache_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                cache[record["key"]] = BoundaryReview(
                    confirmed=record["confirmed"],
                    corrected_page_start=record["corrected_page_start"],
                    corrected_page_end=record["corrected_page_end"],
                    split_suggested=record["split_suggested"],
                    split_notes=record["split_notes"],
                    reasoning=record["reasoning"],
                )
        return cache

    def review(
        self,
        *,
        clause_title: str,
        claimed_page_start: int,
        claimed_page_end: int,
        page_images: tuple[bytes, ...],
    ) -> BoundaryReview:
        """Review, serving from cache when this exact input was seen before."""
        key = _cache_key(
            self._model, clause_title, claimed_page_start, claimed_page_end, page_images
        )
        with self._lock:
            cached = self._cache.get(key)
        if cached is not None:
            return cached

        result = self._inner.review(
            clause_title=clause_title,
            claimed_page_start=claimed_page_start,
            claimed_page_end=claimed_page_end,
            page_images=page_images,
        )

        with self._lock:
            self._cache[key] = result
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self._cache_path.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "key": key,
                            "confirmed": result.confirmed,
                            "corrected_page_start": result.corrected_page_start,
                            "corrected_page_end": result.corrected_page_end,
                            "split_suggested": result.split_suggested,
                            "split_notes": result.split_notes,
                            "reasoning": result.reasoning,
                        }
                    )
                    + "\n"
                )
                f.flush()
        return result
