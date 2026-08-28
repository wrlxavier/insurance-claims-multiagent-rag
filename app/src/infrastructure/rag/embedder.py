"""The embedder interface the [M3-02] index-building pipeline depends on.

A local Protocol, mirroring [infrastructure.evaluation.retriever.Retriever]:
the embedding pipeline is pure infrastructure (a model runtime plus a pgvector
column), so there is no application use case to hang an
``application/ports/`` port off. [M3-04]'s query side will embed queries
through the same contract.

The real implementation (``sentence-transformers`` loading the pinned
[infrastructure.rag.embedding_config] model) is a later [M3-02] slice; the test
suite drives the pipeline with a fake, per the [M1-05b]/[M1-04d] no-live-calls
precedent.
"""

from collections.abc import Sequence
from typing import Protocol


class Embedder(Protocol):
    """Anything that turns text into dense vectors."""

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one vector per input text, in the same order.

        Callers pass strings already run through
        [infrastructure.rag.embedding_config.format_passage] /
        ``format_query`` -- the embedder does no prefixing of its own.
        """
        ...
