"""The deliberately broken retriever this harness validates itself against [M2-06].

Per the M2-06 DoD: a retriever that ignores the question entirely and
returns random clauses should make every metric in
[retrieval_metrics] collapse toward zero. If it doesn't, the harness itself
is broken, not the (nonexistent) retriever.
"""

import random
from collections.abc import Sequence

DEFAULT_SEED = 42


class RandomRetriever:
    """Returns clause ids sampled uniformly at random, ignoring the question."""

    def __init__(self, clause_ids: Sequence[str], *, seed: int = DEFAULT_SEED) -> None:
        """Build the retriever over a fixed pool of clause ids."""
        self._pool = list(clause_ids)
        self._rng = random.Random(seed)

    def retrieve(self, question: str, *, k: int) -> list[str]:
        """Sample min(k, pool size) ids without replacement; question is unused."""
        del question
        return self._rng.sample(self._pool, min(k, len(self._pool)))
