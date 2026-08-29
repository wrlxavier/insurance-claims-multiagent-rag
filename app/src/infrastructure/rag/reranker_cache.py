"""Content-addressed, on-disk cache wrapping a Reranker -- [M3-05].

The ``make tune-reranking`` candidate-depth sweep re-scores heavily overlapping
``(query, clause)`` pairs across depths, and a re-run over the 117 golden
queries should cost nothing. This wraps any
[infrastructure.rag.reranker.Reranker] with a cache keyed by a hash of
(reranker-contract fingerprint, query, passage): a rerun skips every pair
already scored, across separate runs.

The key deliberately covers more than the pair.
[infrastructure.rag.reranker_config.config_fingerprint] folds the model id,
revision, input cap and candidate depth into the key, so a cache built under one
reranker configuration can never silently serve its scores to another -- the
same guard [infrastructure.rag.embedding_cache.CachingEmbedder] applies.

Location and format follow the repo convention
(``data/cache/embeddings/cache.jsonl``,
``data/cache/llm_classification/cache.jsonl``): one gitignored JSON Lines file,
appended on every miss.
"""

import hashlib
import json
import threading
from collections.abc import Sequence
from pathlib import Path

from infrastructure.rag.reranker import Reranker
from infrastructure.rag.reranker_config import config_fingerprint

RERANKER_CACHE_PATH = Path("data/cache/reranker/cache.jsonl")


class CachingReranker:
    """A [Reranker] that persists ``(query, passage) -> score`` to a JSONL file."""

    def __init__(
        self, inner: Reranker, *, cache_path: Path = RERANKER_CACHE_PATH
    ) -> None:
        """Wrap ``inner``, caching its scores under ``cache_path``."""
        self._inner = inner
        self._fingerprint = config_fingerprint()
        self._cache_path = cache_path
        self._lock = threading.Lock()
        self._cache: dict[str, float] = self._load()
        # Per-instance tallies over every pair passed to :meth:`rerank`, so a
        # sweep can report cold-vs-warm behaviour (cold: hits == 0). ``hits +
        # misses`` always equals the number of pairs seen.
        self.hits = 0
        self.misses = 0

    def _key(self, query: str, passage: str) -> str:
        payload = f"{self._fingerprint}\x00{query}\x00{passage}".encode()
        return hashlib.sha256(payload).hexdigest()[:32]

    def _load(self) -> dict[str, float]:
        if not self._cache_path.exists():
            return {}
        cache: dict[str, float] = {}
        with self._cache_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                cache[record["key"]] = record["score"]
        return cache

    def _persist(self, entries: dict[str, float]) -> None:
        self._cache.update(entries)
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self._cache_path.open("a", encoding="utf-8") as f:
            for key, score in entries.items():
                f.write(json.dumps({"key": key, "score": score}) + "\n")
            f.flush()

    def rerank(self, query: str, passages: Sequence[str]) -> list[float]:
        """Return one score per passage, scoring only pairs not already cached."""
        keys = [self._key(query, passage) for passage in passages]
        with self._lock:
            resolved = {key: self._cache[key] for key in keys if key in self._cache}

        cached_keys = set(resolved)
        self.hits += sum(1 for key in keys if key in cached_keys)
        self.misses += sum(1 for key in keys if key not in cached_keys)

        pending: dict[str, str] = {}  # key -> passage, duplicates collapsed
        for passage, key in zip(passages, keys, strict=True):
            if key not in resolved and key not in pending:
                pending[key] = passage

        if pending:
            scores = self._inner.rerank(query, list(pending.values()))
            fresh = dict(zip(pending, scores, strict=True))
            with self._lock:
                self._persist(fresh)
            resolved.update(fresh)

        return [resolved[key] for key in keys]
