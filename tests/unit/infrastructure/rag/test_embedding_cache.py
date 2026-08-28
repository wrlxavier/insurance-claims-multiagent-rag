import hashlib
from collections.abc import Sequence
from pathlib import Path

import pytest

from infrastructure.rag.embedding_cache import CachingEmbedder


def _vec(text: str) -> list[float]:
    """A deterministic, distinct pseudo-vector per text."""
    digest = hashlib.sha256(text.encode()).digest()
    return [digest[0] / 255, digest[1] / 255, digest[2] / 255]


class CountingEmbedder:
    """Records every batch of texts it was asked to embed."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [_vec(text) for text in texts]

    @property
    def call_count(self) -> int:
        return len(self.calls)

    @property
    def embedded(self) -> list[str]:
        return [text for call in self.calls for text in call]


@pytest.mark.unit
def test_cache_miss_calls_inner_and_persists(tmp_path: Path) -> None:
    inner = CountingEmbedder()
    cache_path = tmp_path / "cache.jsonl"

    result = CachingEmbedder(inner, cache_path=cache_path).embed(["cláusula"])

    assert result == [_vec("cláusula")]
    assert inner.call_count == 1
    assert cache_path.exists()
    assert len(cache_path.read_text(encoding="utf-8").strip().splitlines()) == 1


@pytest.mark.unit
def test_cache_hit_skips_inner(tmp_path: Path) -> None:
    inner = CountingEmbedder()
    cache_path = tmp_path / "cache.jsonl"
    embedder = CachingEmbedder(inner, cache_path=cache_path)

    first = embedder.embed(["cláusula"])
    second = embedder.embed(["cláusula"])

    assert first == second == [_vec("cláusula")]
    assert inner.call_count == 1


@pytest.mark.unit
def test_cache_persists_across_instances(tmp_path: Path) -> None:
    cache_path = tmp_path / "cache.jsonl"
    warm_inner = CountingEmbedder()
    CachingEmbedder(warm_inner, cache_path=cache_path).embed(["cláusula", "cobertura"])

    cold_inner = CountingEmbedder()
    result = CachingEmbedder(cold_inner, cache_path=cache_path).embed(
        ["cláusula", "cobertura"]
    )

    assert result == [_vec("cláusula"), _vec("cobertura")]
    assert cold_inner.call_count == 0


@pytest.mark.unit
def test_batch_embeds_only_the_missing_texts_and_preserves_order(
    tmp_path: Path,
) -> None:
    cache_path = tmp_path / "cache.jsonl"
    CachingEmbedder(CountingEmbedder(), cache_path=cache_path).embed(["a", "c"])

    inner = CountingEmbedder()
    result = CachingEmbedder(inner, cache_path=cache_path).embed(["a", "b", "c", "d"])

    assert result == [_vec("a"), _vec("b"), _vec("c"), _vec("d")]
    assert inner.calls == [["b", "d"]]  # only the two not already cached


@pytest.mark.unit
def test_duplicate_texts_in_one_call_embed_once(tmp_path: Path) -> None:
    inner = CountingEmbedder()
    cache_path = tmp_path / "cache.jsonl"

    result = CachingEmbedder(inner, cache_path=cache_path).embed(["x", "x", "x"])

    assert result == [_vec("x"), _vec("x"), _vec("x")]
    assert inner.calls == [["x"]]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("attr", "value"),
    [
        ("EMBEDDING_MODEL_ID", "other/model"),
        ("EMBEDDING_MODEL_REVISION", "0" * 40),
        ("EMBEDDING_DIMENSIONS", 1024),
        ("NORMALIZE_EMBEDDINGS", False),
        ("PASSAGE_PREFIX", "passage: "),
    ],
)
def test_config_change_invalidates_the_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, attr: str, value: object
) -> None:
    cache_path = tmp_path / "cache.jsonl"
    warm_inner = CountingEmbedder()
    CachingEmbedder(warm_inner, cache_path=cache_path).embed(["cláusula"])
    assert warm_inner.call_count == 1

    # Same text, but a different embedding contract -- must NOT reuse the vector.
    monkeypatch.setattr(f"infrastructure.rag.embedding_config.{attr}", value)

    cold_inner = CountingEmbedder()
    CachingEmbedder(cold_inner, cache_path=cache_path).embed(["cláusula"])

    assert cold_inner.embedded == ["cláusula"]
