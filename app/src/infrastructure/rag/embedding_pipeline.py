"""Batched, resumable embedding of the chunk corpus -- [M3-02].

Fills ``chunk.embedding`` for every chunk that has no vector yet. Two properties
the [M3-02] DoD asks for:

* **Batched with retry.** Chunks are embedded ``EMBEDDING_BATCH_SIZE`` at a
  time; each batch's embed call is wrapped in the shared 3-attempt / 5s retry
  ([application.use_cases.llm_retry_defaults]).
* **Resumable cursor.** The cursor is ``WHERE embedding IS NULL``
  ([infrastructure.database.chunk_repository.fetch_chunks_missing_embedding]),
  and each finished batch is committed. A run killed part-way keeps every batch
  it completed, and re-running embeds exactly the remainder -- nothing
  duplicated, nothing skipped.

Why not a stricter/looser retry than [M1-08b]'s: the pinned model runs
in-process (``sentence-transformers``), so there is no API and no rate limit --
the premise behind a retry barely applies. The policy is reused **unchanged**
for uniformity with every other batch job in the repo, and as thin cover for a
transient local failure (a first-call model load, an OOM that clears). What
actually makes an interrupted run safe is the resumable cursor, not the retry.

The real ``Embedder`` is
[infrastructure.rag.sentence_transformer_embedder.SentenceTransformerEmbedder]
(the optional ``embed`` dependency group); ``scripts/embed_chunks.py`` /
``make embed-chunks`` compose it with the cache and run this pipeline, and record
the corpus embedding cost. The test suite drives this module with a fake per the
[M1-05b]/[M1-04d] no-live-calls precedent.
"""

import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from itertools import batched

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from application.use_cases.llm_retry_defaults import (
    DEFAULT_LLM_RETRY_DELAY_SECONDS as EMBEDDING_RETRY_DELAY_SECONDS,
)
from application.use_cases.llm_retry_defaults import (
    DEFAULT_LLM_RETRY_MAX_ATTEMPTS as EMBEDDING_RETRY_MAX_ATTEMPTS,
)
from infrastructure.database.chunk_repository import (
    fetch_chunks_missing_embedding,
    write_chunk_embeddings,
)
from infrastructure.database.models import ChunkRow
from infrastructure.rag.embedder import Embedder
from infrastructure.rag.embedding_config import format_passage

# Pure throughput / memory knob: how many chunks are handed to ``Embedder.embed``
# at once and committed together. No effect on the stored vectors. This is the
# pure-function default; ``scripts/embed_chunks.py`` overrides it from
# ``EmbeddingSettings.embedding_batch_size`` (``.env``'s ``EMBEDDING_BATCH_SIZE``).
# Classified as a ``.env`` knob per the [M1-09] per-constant table in
# docs/EMBEDDINGS.md -- the analog of ``LLM_CLASSIFICATION_MAX_WORKERS``.
EMBEDDING_BATCH_SIZE = 64


@dataclass(frozen=True)
class EmbeddingRun:
    """What one :func:`embed_missing_chunks` call did."""

    embedded: int
    already_present: int
    batches: int


def _embed_with_retry(
    embedder: Embedder,
    texts: Sequence[str],
    *,
    max_attempts: int = EMBEDDING_RETRY_MAX_ATTEMPTS,
    retry_delay_seconds: float = EMBEDDING_RETRY_DELAY_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> list[list[float]]:
    """Call ``embedder.embed``, retrying transient failures, then RE-RAISING.

    Mirrors [application.use_cases.boundary_escalation._review_with_retry]: a
    missing vector has no sane fallback value, unlike the classifier's
    OTHER/0.0, so exhausting the attempts is a hard failure.
    """
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return embedder.embed(texts)
        except Exception as exc:  # retried up to max_attempts, then re-raised
            last_exc = exc
            if attempt < max_attempts:
                sleep(retry_delay_seconds)
    assert last_exc is not None
    raise last_exc


def embed_batches(
    items: Sequence[tuple[str, str]],
    embedder: Embedder,
    *,
    batch_size: int = EMBEDDING_BATCH_SIZE,
    sleep: Callable[[float], None] = time.sleep,
) -> Iterator[dict[str, list[float]]]:
    """Yield ``{chunk_id: vector}`` one batch at a time.

    ``items`` is ``(chunk_id, embedded_text)``. Each text is run through
    [infrastructure.rag.embedding_config.format_passage] before embedding -- the
    single formatting path the index side and [M3-04]'s query side share. Pure:
    no database, no commit.
    """
    for batch in batched(items, batch_size):
        chunk_ids = [chunk_id for chunk_id, _ in batch]
        passages = [format_passage(text) for _, text in batch]
        vectors = _embed_with_retry(embedder, passages, sleep=sleep)
        if len(vectors) != len(chunk_ids):
            raise ValueError(
                f"embedder returned {len(vectors)} vectors for {len(chunk_ids)} texts"
            )
        yield dict(zip(chunk_ids, vectors, strict=True))


def embed_missing_chunks(
    session: Session,
    embedder: Embedder,
    *,
    batch_size: int = EMBEDDING_BATCH_SIZE,
    sleep: Callable[[float], None] = time.sleep,
) -> EmbeddingRun:
    """Embed every chunk still missing a vector, committing per batch.

    The per-batch commit is the resilience mechanism: an interrupted run (a
    kill, or :func:`_embed_with_retry` exhausting its attempts) keeps every
    batch it finished, and a re-run resumes from the rows still at
    ``embedding IS NULL``.
    """
    total: int = session.execute(
        select(func.count()).select_from(ChunkRow)
    ).scalar_one()
    pending = fetch_chunks_missing_embedding(session)

    embedded = 0
    batches = 0
    for vectors in embed_batches(pending, embedder, batch_size=batch_size, sleep=sleep):
        write_chunk_embeddings(session, vectors)
        session.commit()  # the resumable checkpoint -- one per finished batch
        embedded += len(vectors)
        batches += 1

    return EmbeddingRun(
        embedded=embedded,
        already_present=total - len(pending),
        batches=batches,
    )
