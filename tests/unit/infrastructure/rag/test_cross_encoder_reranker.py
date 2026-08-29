"""The real reranker's wiring, proven without installing sentence-transformers.

The optional ``embed`` dependency group is not in the env `make check` / CI run
against, so these tests fake ``sentence_transformers`` in ``sys.modules`` and
never import torch. Mirrors ``test_sentence_transformer_embedder.py``.
"""

import builtins
import sys
import types

import pytest

from infrastructure.rag.cross_encoder_reranker import CrossEncoderReranker
from infrastructure.rag.reranker_config import (
    RERANKER_MODEL_ID,
    RERANKER_MODEL_REVISION,
)


class _FakeCrossEncoder:
    def __init__(
        self,
        model_id: str,
        *,
        revision: str,
        trust_remote_code: bool,
        max_length: int,
        device: str | None,
    ) -> None:
        self.init_kwargs = {
            "model_id": model_id,
            "revision": revision,
            "trust_remote_code": trust_remote_code,
            "max_length": max_length,
        }
        self.device = device or "cpu"
        self.predict_calls: list[list[tuple[str, str]]] = []

    def predict(self, pairs: list[tuple[str, str]], **kwargs: object) -> list[float]:
        self.predict_calls.append(list(pairs))
        # Deterministic, distinct score per passage: longer passage scores higher.
        return [float(len(passage)) for _query, passage in pairs]


def _install_fake(monkeypatch: pytest.MonkeyPatch) -> list[_FakeCrossEncoder]:
    created: list[_FakeCrossEncoder] = []

    def factory(
        model_id: str,
        *,
        revision: str,
        trust_remote_code: bool,
        max_length: int,
        device: str | None,
    ) -> _FakeCrossEncoder:
        model = _FakeCrossEncoder(
            model_id,
            revision=revision,
            trust_remote_code=trust_remote_code,
            max_length=max_length,
            device=device,
        )
        created.append(model)
        return model

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(CrossEncoder=factory),
    )
    return created


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
        CrossEncoderReranker()


@pytest.mark.unit
def test_loads_the_pinned_model_and_scores_pairs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created = _install_fake(monkeypatch)

    reranker = CrossEncoderReranker(device="cpu")

    assert created[0].init_kwargs == {
        "model_id": RERANKER_MODEL_ID,
        "revision": RERANKER_MODEL_REVISION,
        "trust_remote_code": True,
        "max_length": 8192,
    }
    assert reranker.device == "cpu"

    scores = reranker.rerank("qual a franquia?", ["curto", "um trecho bem mais longo"])

    assert scores == [5.0, 24.0]
    assert created[0].predict_calls[0] == [
        ("qual a franquia?", "curto"),
        ("qual a franquia?", "um trecho bem mais longo"),
    ]


@pytest.mark.unit
def test_empty_passages_never_touches_the_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created = _install_fake(monkeypatch)

    assert CrossEncoderReranker().rerank("q", []) == []
    assert created[0].predict_calls == []
