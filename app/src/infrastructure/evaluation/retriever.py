"""The retrieval interface this evaluation harness scores [M2-06].

Decoupled from ``infrastructure.rag`` -- M3's real hybrid retriever doesn't
exist yet (see ``MILESTONES.md``, M3 is ``todo``) -- so this harness can be
validated now, against [random_retriever.RandomRetriever]'s deliberately
broken implementation, and a future M3 retriever can be pointed at the same
harness unmodified by implementing this one method.
"""

from typing import Protocol


class Retriever(Protocol):
    """Anything that can rank clause ids against a question's text."""

    def retrieve(self, question: str, *, k: int) -> list[str]:
        """Return up to ``k`` clause ids, ranked best-match first."""
        ...
