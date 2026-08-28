"""Unit tests for the batched embedding pipeline -- [M3-02].

No live model: the embedder is a hand-written fake, constructor-injected,
matching the fake-chat-model precedent in the [M1-05b]/[M1-04d] tests. The
database-backed half (the resumable cursor, the write path, the interrupt/
resume proof) is covered in tests/integration/test_chunk_embedding.py.
"""

import hashlib
import math
from collections.abc import Sequence

import pytest

from application.use_cases.llm_retry_defaults import DEFAULT_LLM_RETRY_MAX_ATTEMPTS
from infrastructure.rag.embedding_config import EMBEDDING_DIMENSIONS, format_passage
from infrastructure.rag.embedding_pipeline import _embed_with_retry, embed_batches


def _unit_vector(text: str) -> list[float]:
    """A deterministic, L2-normalised pseudo-embedding of the right width."""
    digest = hashlib.sha256(text.encode()).digest()
    raw = [digest[i % len(digest)] + 1 for i in range(EMBEDDING_DIMENSIONS)]
    norm = math.sqrt(sum(value * value for value in raw))
    return [value / norm for value in raw]


class FakeEmbedder:
    """Records every ``embed`` call; returns a deterministic vector per text."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [_unit_vector(text) for text in texts]


class FlakyEmbedder:
    """Raises ``RuntimeError`` for the first ``fail_times`` calls, then delegates."""

    def __init__(self, *, fail_times: int) -> None:
        self._remaining = fail_times
        self._inner = FakeEmbedder()
        self.calls: list[list[str]] = []

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        if self._remaining > 0:
            self._remaining -= 1
            raise RuntimeError("transient embed failure")
        return self._inner.embed(texts)


class ShortEmbedder:
    """Returns one fewer vector than asked -- a contract violation to catch."""

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [_unit_vector(text) for text in texts][:-1]


_ITEMS = [(f"1:{i}", f"clause text {i}") for i in range(5)]


@pytest.mark.unit
def test_embed_batches_respects_batch_size() -> None:
    embedder = FakeEmbedder()

    list(embed_batches(_ITEMS, embedder, batch_size=2))

    assert [len(call) for call in embedder.calls] == [2, 2, 1]


@pytest.mark.unit
def test_embed_batches_covers_every_chunk_exactly_once() -> None:
    embedder = FakeEmbedder()

    seen: dict[str, list[float]] = {}
    for batch in embed_batches(_ITEMS, embedder, batch_size=2):
        for chunk_id, vector in batch.items():
            assert chunk_id not in seen  # nothing duplicated across batches
            seen[chunk_id] = vector

    assert set(seen) == {chunk_id for chunk_id, _ in _ITEMS}
    assert all(len(vector) == EMBEDDING_DIMENSIONS for vector in seen.values())


@pytest.mark.unit
def test_embed_batches_embeds_the_formatted_passage() -> None:
    embedder = FakeEmbedder()

    list(embed_batches(_ITEMS, embedder, batch_size=10))

    # `format_passage` is the one formatting path the index and query sides
    # share; assert the embedder sees its output, not the raw text.
    assert embedder.calls == [[format_passage(text) for _, text in _ITEMS]]


@pytest.mark.unit
def test_embed_batches_rejects_a_wrong_vector_count() -> None:
    with pytest.raises(ValueError, match="4 vectors for 5 texts"):
        list(embed_batches(_ITEMS, ShortEmbedder(), batch_size=5))


@pytest.mark.unit
def test_retry_succeeds_after_transient_failures() -> None:
    slept: list[float] = []
    embedder = FlakyEmbedder(fail_times=2)

    vectors = _embed_with_retry(embedder, ["a", "b"], sleep=slept.append)

    assert len(vectors) == 2
    assert len(embedder.calls) == 3  # two failures, then success
    assert slept == [5.0, 5.0]


@pytest.mark.unit
def test_retry_reraises_after_exhausting_attempts() -> None:
    slept: list[float] = []
    embedder = FlakyEmbedder(fail_times=99)

    with pytest.raises(RuntimeError, match="transient embed failure"):
        _embed_with_retry(embedder, ["a"], sleep=slept.append)

    assert len(embedder.calls) == DEFAULT_LLM_RETRY_MAX_ATTEMPTS
    assert slept == [5.0] * (DEFAULT_LLM_RETRY_MAX_ATTEMPTS - 1)


@pytest.mark.unit
def test_embed_batches_propagates_an_exhausted_retry() -> None:
    with pytest.raises(RuntimeError, match="transient embed failure"):
        list(
            embed_batches(
                _ITEMS,
                FlakyEmbedder(fail_times=99),
                batch_size=2,
                sleep=lambda _: None,
            )
        )
