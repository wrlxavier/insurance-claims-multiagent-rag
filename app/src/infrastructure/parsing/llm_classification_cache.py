"""Content-addressed, on-disk cache wrapping a ClauseClassifierPort.

``scripts/build_corpus.py``'s LLM pass is the slowest and only non-free
stage of ``make parse`` -- a kill or crash partway through should not mean
re-classifying (and re-paying for) everything from scratch. This wraps any
[application.ports.clause_classifier.ClauseClassifierPort] with a cache
keyed by a hash of (model, title, text): a rerun after a kill skips every
clause already classified, and identical clause content is never
reclassified even across separate runs.
"""

import hashlib
import json
import threading
from pathlib import Path

from application.ports.clause_classifier import ClauseClassifierPort
from domain.clause_classification import ClauseType


def _cache_key(model: str, clause_title: str, clause_text: str) -> str:
    payload = f"{model}\x00{clause_title}\x00{clause_text}".encode()
    return hashlib.sha256(payload).hexdigest()[:32]


class CachingClauseClassifier:
    """A [ClauseClassifierPort] that persists results to a JSON Lines file."""

    def __init__(
        self, inner: ClauseClassifierPort, model: str, cache_path: Path
    ) -> None:
        """Wrap ``inner``, caching its results under ``cache_path``."""
        self._inner = inner
        self._model = model
        self._cache_path = cache_path
        self._lock = threading.Lock()
        self._cache: dict[str, tuple[ClauseType, float]] = self._load()

    def _load(self) -> dict[str, tuple[ClauseType, float]]:
        if not self._cache_path.exists():
            return {}
        cache: dict[str, tuple[ClauseType, float]] = {}
        with self._cache_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                cache[record["key"]] = (
                    ClauseType(record["clause_type"]),
                    record["confidence"],
                )
        return cache

    def classify(self, clause_title: str, clause_text: str) -> tuple[ClauseType, float]:
        """Classify, serving from cache when this exact input was seen before."""
        key = _cache_key(self._model, clause_title, clause_text)
        with self._lock:
            cached = self._cache.get(key)
        if cached is not None:
            return cached

        result = self._inner.classify(clause_title, clause_text)

        with self._lock:
            self._cache[key] = result
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self._cache_path.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "key": key,
                            "clause_type": result[0].value,
                            "confidence": result[1],
                        }
                    )
                    + "\n"
                )
                f.flush()
        return result
