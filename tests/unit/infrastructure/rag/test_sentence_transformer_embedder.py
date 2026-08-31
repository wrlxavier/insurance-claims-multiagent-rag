"""The real embedder's wiring, proven without installing sentence-transformers.

The optional ``embed`` dependency group is not in the env `make check` / CI run
against, so these tests fake ``sentence_transformers`` in ``sys.modules`` and
never import torch.
"""

import builtins
import sys
import types
from collections.abc import Callable, Sequence

import pytest

from infrastructure.rag.embedding_config import (
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL_ID,
    EMBEDDING_MODEL_REVISION,
)
from infrastructure.rag.sentence_transformer_embedder import SentenceTransformerEmbedder


class _FakeModel:
    def __init__(
        self,
        model_id: str,
        *,
        revision: str,
        trust_remote_code: bool,
        device: str | None,
        width: int = EMBEDDING_DIMENSIONS,
    ) -> None:
        self.init_kwargs = {
            "model_id": model_id,
            "revision": revision,
            "trust_remote_code": trust_remote_code,
        }
        self.device = device or "cpu"
        self.width = width
        self.encode_calls: list[tuple[list[str], dict[str, object]]] = []

    def encode(self, texts: Sequence[str], **kwargs: object) -> list[list[float]]:
        self.encode_calls.append((list(texts), kwargs))
        return [[0.1] * self.width for _ in texts]


_Factory = Callable[..., _FakeModel]


def _install_fake(monkeypatch: pytest.MonkeyPatch, factory: _Factory) -> None:
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(SentenceTransformer=factory),
    )


@pytest.mark.unit
def test_missing_dependency_raises_a_pointed_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "sentence_transformers" or name.startswith("sentence_transformers."):
            raise ModuleNotFoundError("No module named 'sentence_transformers'")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.delitem(sys.modules, "sentence_transformers", raising=False)
    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ModuleNotFoundError, match="--group embed"):
        SentenceTransformerEmbedder()


@pytest.mark.unit
def test_loads_the_pinned_model_and_embeds_normalised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[_FakeModel] = []

    def factory(
        model_id: str, *, revision: str, trust_remote_code: bool, device: str | None
    ) -> _FakeModel:
        model = _FakeModel(
            model_id,
            revision=revision,
            trust_remote_code=trust_remote_code,
            device=device,
        )
        created.append(model)
        return model

    _install_fake(monkeypatch, factory)

    embedder = SentenceTransformerEmbedder(batch_size=8, device="cpu")

    assert created[0].init_kwargs == {
        "model_id": EMBEDDING_MODEL_ID,
        "revision": EMBEDDING_MODEL_REVISION,
        "trust_remote_code": True,
    }
    assert embedder.device == "cpu"

    vectors = embedder.embed(["cláusula", "cobertura"])

    assert vectors == [[0.1] * EMBEDDING_DIMENSIONS] * 2
    texts, kwargs = created[0].encode_calls[0]
    assert texts == ["cláusula", "cobertura"]
    assert kwargs["normalize_embeddings"] is True
    assert kwargs["batch_size"] == 8


@pytest.mark.unit
def test_empty_input_never_touches_the_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[_FakeModel] = []

    def factory(
        model_id: str, *, revision: str, trust_remote_code: bool, device: str | None
    ) -> _FakeModel:
        model = _FakeModel(
            model_id,
            revision=revision,
            trust_remote_code=trust_remote_code,
            device=device,
        )
        created.append(model)
        return model

    _install_fake(monkeypatch, factory)

    assert SentenceTransformerEmbedder().embed([]) == []
    assert created[0].encode_calls == []


@pytest.mark.unit
def test_wrong_width_vector_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    def factory(
        model_id: str, *, revision: str, trust_remote_code: bool, device: str | None
    ) -> _FakeModel:
        return _FakeModel(
            model_id,
            revision=revision,
            trust_remote_code=trust_remote_code,
            device=device,
            width=EMBEDDING_DIMENSIONS - 1,
        )

    _install_fake(monkeypatch, factory)

    with pytest.raises(ValueError, match=f"expected {EMBEDDING_DIMENSIONS}"):
        SentenceTransformerEmbedder().embed(["x"])
