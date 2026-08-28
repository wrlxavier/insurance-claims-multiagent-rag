"""The retrieval interface this evaluation harness scores [M2-06].

Decoupled from ``infrastructure.rag`` -- M3's real hybrid retriever doesn't
exist yet (see ``MILESTONES.md``, M3 is ``todo``) -- so this harness can be
validated now, against [random_retriever.RandomRetriever]'s deliberately
broken implementation, and a future M3 retriever can be pointed at the same
harness unmodified by implementing this one method.

[M3-04] adds [FilterableRetriever]: the same contract plus an optional
``metadata_filter``. The dense, lexical and hybrid retrievers implement it;
``random`` implements only the bare [Retriever]. The ``RetrievalFilter``
reference is type-checking only, so importing this module still pulls in no
``infrastructure.rag`` code.
"""

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from infrastructure.rag.retrieval_filter import RetrievalFilter


class Retriever(Protocol):
    """Anything that can rank clause ids against a question's text."""

    def retrieve(self, question: str, *, k: int) -> list[str]:
        """Return up to ``k`` clause ids, ranked best-match first."""
        ...


class FilterableRetriever(Protocol):
    """A [Retriever] that also honours a [M3-04] metadata pre-filter."""

    def retrieve(
        self,
        question: str,
        *,
        k: int,
        metadata_filter: "RetrievalFilter | None" = None,
    ) -> list[str]:
        """Return up to ``k`` clause ids, ranked best-match first, filtered."""
        ...
