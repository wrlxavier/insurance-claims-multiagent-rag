"""The pinned insufficient-context gate contract -- [M3-07].

Single source of truth for the retrieval-signal thresholds that decide whether
the retrieved context settles a question or the system abstains with
``insufficient_information``. The [M3-07] calibration harness
(``scripts/eval_insufficient_context_gate.py``) and M4's retrieval node
([M4-04], which "sets the insufficient-context flag from the [M3-07] gate")
both reach the gate through this module, so the thresholds a published
gate-precision number was measured under and the ones M4 runs cannot drift
apart. Recorded here, as code, not in a comment.

Why these are module constants and not ``.env`` knobs: the thresholds
determine the exact precision/recall the M3 exit criterion is measured against
("gate precision on the unanswerable subset ... reference value >= 80%", "gate
recall ... 100%", ``MILESTONES.md``). Per [M1-09]'s per-constant rule a value
that moves a published number is experimental design -- like
``RERANK_CANDIDATE_DEPTH`` in ``reranker_config.py``, ``RESERVED_EXCLUSION_SLOTS``
in ``exclusion_co_retrieval_config.py`` or ``SEED`` in
``scripts/sample_parsing_quality.py`` -- and making one environment-dependent
would let a ``.env`` edit silently move a published result. This issue
introduces **no** new ``.env`` key -- the analog of ``docs/RERANKING.md``'s
"zero" and ``docs/HYBRID_RETRIEVAL.md``'s "zero".

Rationale, calibration method, sweep and numbers: docs/INSUFFICIENT_CONTEXT_GATE.md.
"""

import hashlib

# The reranker (``Alibaba-NLP/gte-multilingual-reranker-base``) emits a sigmoid
# relevance score in [0, 1]. The gate reads its two thresholds off that scale.

# Primary floor: abstain whenever the rank-1 reranked clause scores below this.
# A poor topical match means retrieval found nothing that could answer. Set from
# the ``make eval-insufficient-context-gate`` sweep
# (docs/INSUFFICIENT_CONTEXT_GATE.md): the lowest answerable ``golden-set-v1``
# top-score is 0.483, the highest unanswerable top-score caught by this floor
# alone is 0.450 -- 0.46 sits in that gap. A code constant: it determines the
# published gate precision/recall.
TOP_SCORE_ABSTAIN_THRESHOLD = 0.46

# Strict floor for a question that asks for a *specific* policy-instance value of
# a fact type ``docs/SCOPE.md`` says the corpus does not contain (a deductible /
# insured-amount / premium amount, a policy period, an endorsement) -- see
# ``insufficient_context_gate.needs_verified_instance_value``. For those, a
# topically-strong clause that merely *discusses* the fact (a calculation
# formula, a general rule) is the exact failure mode the gate guards against, so
# retrieval must clear a much higher bar before the context is trusted. Set from
# the same sweep: the 14 instance-value unanswerable questions top out at 0.830;
# the 2 answerable questions that also ask for such a value score 0.883 / 0.966
# (their answer really is in the corpus and the reranker finds it cleanly) --
# 0.84 sits in that gap. A code constant, same reason as above.
INSTANCE_VALUE_TOP_SCORE_THRESHOLD = 0.84


def config_fingerprint() -> str:
    """Digest of every value that determines the gate's abstain decision.

    Stamped into the ``make eval-insufficient-context-gate`` run's
    ``InsufficientContextGateConfig`` and the committed signal snapshot
    (``eval/insufficient_context_gate_signals.json``), so a report or a snapshot
    read in isolation says exactly which thresholds produced it -- and M4-04 can
    record which gate contract it routed on.

    Reads the module constants at call time (not import time) so a test can
    monkeypatch one and watch the digest move -- mirrors the sibling
    ``config_fingerprint``s in ``reranker_config`` / ``hybrid_config`` /
    ``embedding_config`` / ``lexical_config`` / ``exclusion_co_retrieval_config``.
    """
    payload = "\x00".join(
        (
            str(TOP_SCORE_ABSTAIN_THRESHOLD),
            str(INSTANCE_VALUE_TOP_SCORE_THRESHOLD),
        )
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]
