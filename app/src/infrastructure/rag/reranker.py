"""The cross-encoder interface the [M3-05] reranking stage depends on.

A local Protocol, mirroring [infrastructure.rag.embedder.Embedder] and
[infrastructure.evaluation.retriever.Retriever]: reranking is pure
infrastructure (a model runtime that scores a query against a passage), so
there is no application use case to hang an ``application/ports/`` port off.

The real implementation is
[infrastructure.rag.cross_encoder_reranker.CrossEncoderReranker]
(``sentence-transformers`` loading the pinned
[infrastructure.rag.reranker_config] model, from the optional ``embed``
dependency group). The test suite drives the reranking retriever with a fake,
per the [M1-05b]/[M1-04d] no-live-calls precedent.
"""

from collections.abc import Sequence
from typing import Protocol


class Reranker(Protocol):
    """Anything that scores a query against each of several passages."""

    def rerank(self, query: str, passages: Sequence[str]) -> list[float]:
        """Return one relevance score per passage, in input order.

        Higher means more relevant. The scale is model-defined and only its
        ordering is used -- [infrastructure.rag.reranking_retriever.
        RerankingRetriever] sorts by it and discards the values.
        """
        ...
