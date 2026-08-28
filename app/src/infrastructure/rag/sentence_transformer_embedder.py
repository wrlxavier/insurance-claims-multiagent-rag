"""The real, local :class:`~infrastructure.rag.embedder.Embedder` -- [M3-02].

Loads the pinned ``Alibaba-NLP/gte-multilingual-base`` (see
[infrastructure.rag.embedding_config]) with ``sentence-transformers`` and embeds
in-process: no API, no rate limit, no per-token charge.

``sentence-transformers`` (and its ``torch`` / ``transformers`` dependencies) is
the optional ``embed`` dependency group, deliberately kept out of ``uv sync`` and
CI. Import this module freely -- the heavy import is deferred to
``SentenceTransformerEmbedder.__init__``; only constructing one needs the group
installed (``make embed-chunks`` runs ``uv run --group embed``).
"""

from __future__ import annotations

from collections.abc import Sequence

from infrastructure.rag.embedding_config import (
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL_ID,
    EMBEDDING_MODEL_REVISION,
    EMBEDDING_TRUST_REMOTE_CODE,
    NORMALIZE_EMBEDDINGS,
)
from infrastructure.rag.embedding_pipeline import EMBEDDING_BATCH_SIZE


class SentenceTransformerEmbedder:
    """Embed passages/queries with the pinned model, loaded once at construction."""

    def __init__(
        self,
        *,
        batch_size: int = EMBEDDING_BATCH_SIZE,
        device: str | None = None,
    ) -> None:
        """Load the pinned model at its pinned revision.

        ``device`` is passed straight to ``sentence-transformers`` -- ``None``
        auto-selects (CUDA if available, else CPU); pass ``"cpu"`` to force it.
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ModuleNotFoundError as exc:  # pragma: no cover - env-dependent
            raise ModuleNotFoundError(
                "sentence-transformers is not installed. It is the optional "
                "`embed` dependency group: run `make embed-chunks` (which uses "
                "`uv run --group embed`) or `uv sync --group embed` first."
            ) from exc

        self._batch_size = batch_size
        self._model = SentenceTransformer(
            EMBEDDING_MODEL_ID,
            revision=EMBEDDING_MODEL_REVISION,
            trust_remote_code=EMBEDDING_TRUST_REMOTE_CODE,
            device=device,
        )

    @property
    def device(self) -> str:
        """The torch device the model ended up on, e.g. ``"cpu"`` / ``"cuda:0"``."""
        return str(self._model.device)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one L2-normalised vector per text, in input order."""
        if not texts:
            return []
        raw = self._model.encode(
            list(texts),
            batch_size=self._batch_size,
            normalize_embeddings=NORMALIZE_EMBEDDINGS,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        vectors: list[list[float]] = [[float(value) for value in row] for row in raw]
        bad = [len(vector) for vector in vectors if len(vector) != EMBEDDING_DIMENSIONS]
        if bad:
            raise ValueError(
                f"model returned {bad[0]}-dim vector(s), expected "
                f"{EMBEDDING_DIMENSIONS}"
            )
        return vectors
