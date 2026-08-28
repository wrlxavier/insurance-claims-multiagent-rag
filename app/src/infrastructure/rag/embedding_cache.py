"""Content-addressed, on-disk cache wrapping an Embedder -- [M3-02].

Re-running the embedding pipeline over an unchanged corpus should cost nothing.
This wraps any [infrastructure.rag.embedder.Embedder] with a cache keyed by a
hash of (embedding-contract fingerprint, input text): a rerun skips every text
already embedded, across separate runs.

The key deliberately covers more than the text.
[infrastructure.rag.embedding_config.config_fingerprint] folds the model id,
revision, dimensionality, distance metric, normalisation and both prefixes into
the key, so a cache built under one model configuration can never silently serve
its vectors to another -- the DoD's "most expensive failure mode".

Location and format follow the repo convention
(``data/cache/llm_classification/cache.jsonl``,
``data/cache/boundary_escalation/cache.jsonl``): one gitignored JSON Lines file,
appended on every miss. See docs/EMBEDDINGS.md for why the ~4,540-chunk corpus's
volume does not justify departing from it.

``scripts/embed_chunks.py`` / ``make embed-chunks`` wrap the real
[infrastructure.rag.sentence_transformer_embedder.SentenceTransformerEmbedder] in
a ``CachingEmbedder`` and pass it to
[infrastructure.rag.embedding_pipeline.embed_missing_chunks], mirroring how
``scripts/build_corpus.py`` wraps its classifier in
[infrastructure.parsing.llm_classification_cache.CachingClauseClassifier].
"""

import hashlib
import json
import threading
from collections.abc import Sequence
from pathlib import Path

from infrastructure.rag.embedder import Embedder
from infrastructure.rag.embedding_config import config_fingerprint

EMBEDDING_CACHE_PATH = Path("data/cache/embeddings/cache.jsonl")


class CachingEmbedder:
    """An [Embedder] that persists vectors to a JSON Lines file."""

    def __init__(
        self, inner: Embedder, *, cache_path: Path = EMBEDDING_CACHE_PATH
    ) -> None:
        """Wrap ``inner``, caching its vectors under ``cache_path``."""
        self._inner = inner
        self._fingerprint = config_fingerprint()
        self._cache_path = cache_path
        self._lock = threading.Lock()
        self._cache: dict[str, list[float]] = self._load()
        # Per-instance tallies over every text passed to :meth:`embed`, so
        # ``scripts/embed_chunks.py`` can report a cold run's cache behaviour
        # (cold: hits == 0, misses == chunk count). ``hits + misses`` always
        # equals the number of texts seen.
        self.hits = 0
        self.misses = 0

    def _key(self, text: str) -> str:
        payload = f"{self._fingerprint}\x00{text}".encode()
        return hashlib.sha256(payload).hexdigest()[:32]

    def _load(self) -> dict[str, list[float]]:
        if not self._cache_path.exists():
            return {}
        cache: dict[str, list[float]] = {}
        with self._cache_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                cache[record["key"]] = record["vector"]
        return cache

    def _persist(self, entries: dict[str, list[float]]) -> None:
        self._cache.update(entries)
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self._cache_path.open("a", encoding="utf-8") as f:
            for key, vector in entries.items():
                f.write(json.dumps({"key": key, "vector": vector}) + "\n")
            f.flush()

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one vector per text, embedding only those not already cached."""
        keys = [self._key(text) for text in texts]
        with self._lock:
            resolved = {key: self._cache[key] for key in keys if key in self._cache}

        cached_keys = set(resolved)
        self.hits += sum(1 for key in keys if key in cached_keys)
        self.misses += sum(1 for key in keys if key not in cached_keys)

        pending: dict[str, str] = {}  # key -> text, duplicates collapsed
        for text, key in zip(texts, keys, strict=True):
            if key not in resolved and key not in pending:
                pending[key] = text

        if pending:
            vectors = self._inner.embed(list(pending.values()))
            fresh = dict(zip(pending, vectors, strict=True))
            with self._lock:
                self._persist(fresh)
            resolved.update(fresh)

        return [resolved[key] for key in keys]
