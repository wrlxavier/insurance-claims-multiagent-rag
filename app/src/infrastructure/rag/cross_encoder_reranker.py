"""The real, local :class:`~infrastructure.rag.reranker.Reranker` -- [M3-05].

Loads the pinned ``Alibaba-NLP/gte-multilingual-reranker-base`` (see
[infrastructure.rag.reranker_config]) with ``sentence-transformers``'
``CrossEncoder`` and scores ``(query, passage)`` pairs in-process: no API, no
rate limit, no per-token charge.

``sentence-transformers`` (and its ``torch`` / ``transformers`` dependencies) is
the optional ``embed`` dependency group, deliberately kept out of ``uv sync``
and CI -- the same group the embedder uses, and it already ships ``CrossEncoder``
(no new dependency). Import this module freely: the heavy import is deferred to
``CrossEncoderReranker.__init__``; only constructing one needs the group
installed (``make eval-retrieval-rerank`` / ``make tune-reranking`` run
``uv run --group embed``).
"""

from __future__ import annotations

from collections.abc import Sequence

from infrastructure.rag.reranker_config import (
    RERANKER_MAX_INPUT_TOKENS,
    RERANKER_MODEL_ID,
    RERANKER_MODEL_REVISION,
    RERANKER_TRUST_REMOTE_CODE,
)

# Pairs scored per forward batch. A pure throughput / memory lever -- no effect
# on the scores or their order (padding is attention-masked), so (per [M1-09]) it
# is a plain module constant, not a `.env` knob and not part of
# ``reranker_config.config_fingerprint()``. Held at 1, not a library-typical
# 32/64: a batch pads to its longest member, and one 8192-token clause (the input
# cap) already fills the activation budget of a 4 GB dev GPU; a batch of 64 of
# them needs >12 GB, past a 14 GB CPU box too. Per-pair cost is dominated by the
# model forward, so 1 costs little (GPU: ~45 ms/pair; CPU is slow either way) and
# bounds the peak on any hardware. M4 sets its own value for its serving box.
# See docs/RERANKING.md.
RERANK_BATCH_SIZE = 1


class CrossEncoderReranker:
    """Score query/passage pairs with the pinned cross-encoder, loaded once."""

    def __init__(
        self,
        *,
        batch_size: int = RERANK_BATCH_SIZE,
        max_length: int = RERANKER_MAX_INPUT_TOKENS,
        device: str | None = None,
    ) -> None:
        """Load the pinned model at its pinned revision.

        ``device`` is passed straight to ``sentence-transformers`` -- ``None``
        auto-selects (CUDA if available, else CPU); pass ``"cpu"`` to force it.
        """
        try:
            from sentence_transformers import CrossEncoder
        except ModuleNotFoundError as exc:  # pragma: no cover - env-dependent
            raise ModuleNotFoundError(
                "sentence-transformers is not installed. It is the optional "
                "`embed` dependency group: run `make eval-retrieval-rerank` "
                "(which uses `uv run --group embed`) or `uv sync --group embed` "
                "first."
            ) from exc

        self._batch_size = batch_size
        self._model = CrossEncoder(
            RERANKER_MODEL_ID,
            revision=RERANKER_MODEL_REVISION,
            trust_remote_code=RERANKER_TRUST_REMOTE_CODE,
            max_length=max_length,
            device=device,
        )

    @property
    def device(self) -> str:
        """The torch device the model ended up on, e.g. ``"cpu"`` / ``"cuda:0"``."""
        for attr in ("device", "_target_device"):
            value = getattr(self._model, attr, None)
            if value is not None:
                return str(value)
        return "unknown"

    def rerank(self, query: str, passages: Sequence[str]) -> list[float]:
        """Return one relevance score per passage, in input order.

        Higher is more relevant. Only the ordering is consumed downstream
        ([infrastructure.rag.reranking_retriever.RerankingRetriever]).
        """
        if not passages:
            return []
        scores = self._model.predict(
            [(query, passage) for passage in passages],
            batch_size=self._batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [float(score) for score in scores]
