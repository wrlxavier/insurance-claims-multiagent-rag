"""The pinned hybrid-retrieval contract -- [M3-04].

Single source of truth for the fusion strategy and its parameters: the RRF
smoothing constant, the weighted-fusion weights, how deep each leg is sampled
before fusion, and which strategy is the default. The [M3-04] eval
(``docs/HYBRID_RETRIEVAL.md``) compares RRF against weighted score fusion on the
golden set and sets ``DEFAULT_FUSION_STRATEGY`` to the winner.

Why these are module constants and not ``.env`` knobs: RRF ``k``, the weights
and the candidate depth all change the published Recall@k / MRR. Per [M1-09]'s
per-constant rule they are experimental design -- like ``SEED`` / ``SAMPLE_SIZE``
in ``scripts/sample_parsing_quality.py`` -- and making them environment-dependent
would let a ``.env`` edit silently move a published result. This issue
introduces **no** ``.env`` key (the analog of docs/LEXICAL_RETRIEVAL.md's "zero"
and docs/EMBEDDINGS.md's "exactly one").

Rationale and evidence: docs/HYBRID_RETRIEVAL.md.
"""

import hashlib
from enum import StrEnum

from infrastructure.rag.embedding_config import config_fingerprint as _dense_fingerprint


class FusionStrategy(StrEnum):
    """How the lexical and dense legs are combined."""

    RRF = "rrf"
    WEIGHTED = "weighted"


# Reciprocal Rank Fusion smoothing constant. 60 is the value from Cormack et
# al.'s original RRF paper and the de-facto default across IR toolkits; tuning
# it is [M3-08]'s, not this issue's. It changes the published Recall@k, so it
# stays a code constant.
RRF_K = 60

# (lexical, dense) weights for weighted score fusion, positional and matched to
# the leg order in [infrastructure.rag.hybrid_retriever.HybridRetriever]. Start
# balanced; the [M3-04] eval records whether a tilt helps.
FUSION_WEIGHTS: tuple[float, float] = (0.5, 0.5)

# How many clause ids each leg contributes before fusion. Deep enough that a
# clause outside either leg's top 10 can still be rescued by cross-leg
# agreement, shallow enough to stay cheap at ~4.9k clauses. Changes the fused
# ranking, so a code constant.
CANDIDATE_DEPTH = 100

# Set from the docs/HYBRID_RETRIEVAL.md comparison: RRF unless weighted wins
# filtered overall Recall@10 or MRR by a clear margin.
DEFAULT_FUSION_STRATEGY = FusionStrategy.RRF


def config_fingerprint(*, lexical_config_fingerprint: str) -> str:
    """Digest of every value that determines the fused ranking.

    Folds in the caller-supplied lexical fingerprint
    ([infrastructure.rag.lexical_config.config_fingerprint], which needs the
    on-disk exception list) and computes the dense one itself
    ([infrastructure.rag.embedding_config.config_fingerprint]), so a change to
    the BM25 analyzer or the embedding model moves this too. Threaded into
    [infrastructure.evaluation.retrieval_run_schema.RetrievalRunConfig] so a
    report read in isolation says exactly what produced its numbers. Reads the
    module constants at call time so a test can monkeypatch one and watch the
    digest move (mirrors the sibling ``config_fingerprint``s).
    """
    payload = "\x00".join(
        (
            str(RRF_K),
            ",".join(repr(weight) for weight in FUSION_WEIGHTS),
            str(CANDIDATE_DEPTH),
            DEFAULT_FUSION_STRATEGY.value,
            lexical_config_fingerprint,
            _dense_fingerprint(),
        )
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]
