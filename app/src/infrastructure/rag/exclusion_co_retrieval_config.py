"""The pinned exclusion co-retrieval contract -- [M3-06].

Single source of truth for the post-ranking step that guarantees the exclusion
clauses limiting a retrieved coverage clause survive into the final context. The
[M2-06] eval harness (query side) and M4's retrieval node both reach the step
through this module, so the ranking a golden-set number was measured under and
the one M4 runs cannot drift apart. Recorded here, as code, not in a comment.

Why these are module constants and not ``.env`` knobs: the reserved-slot count
and the adjacent-section page gap together determine the exact top-k the
published Recall@k / exclusion-clause recall are measured on. Per [M1-09]'s
per-constant rule they are experimental design -- like ``RERANK_CANDIDATE_DEPTH``
in ``reranker_config.py`` or ``SEED`` in ``scripts/sample_parsing_quality.py`` --
and making them environment-dependent would let a ``.env`` edit silently move a
published result. This issue introduces **no** new ``.env`` key -- the analog of
``docs/RERANKING.md``'s "zero" and ``docs/HYBRID_RETRIEVAL.md``'s "zero".

Rationale and evidence: docs/EXCLUSION_CO_RETRIEVAL.md. The domain rule the
mechanism serves: docs/ARCHITECTURE.md.
"""

import hashlib

# How many of the final top-k slots are reserved for exclusion clauses linked to
# a retrieved coverage clause, so a higher-scoring coverage passage cannot
# squeeze the exclusion that limits it out of the context ([M3-06] DoD: "reserve
# context budget for exclusions"). Only ever consumed when a retrieved coverage
# clause has a best-linked exclusion the base ranking missed -- otherwise the
# output is the base ranking unchanged. Set to 1 from the ``make
# tune-exclusion-co-retrieval`` sweep in docs/EXCLUSION_CO_RETRIEVAL.md: the
# hybrid + rerank base already surfaces exclusions well, so one reserved slot
# recovers the residual misses (exclusion-clause recall 25/27 -> 27/27) with no
# collateral regression, while 2+ start displacing relevant lookup answers. A
# code constant: it changes the top-k the golden-set numbers are measured on.
RESERVED_EXCLUSION_SLOTS = 1

# The adjacent-section edge fires for an exclusion clause in a *different*
# top-level section of the same document whose page span is within this many
# pages of the coverage clause's. Calibrated on the golden set's flat /
# parent-less OCR documents (doc 4: the reference exclusion sits <= 1 page from
# its coverage clause; doc 7: 2 pages), where the same-section edge structurally
# cannot fire because every clause is its own section root. A code constant for
# the same reason as above.
ADJACENT_SECTION_MAX_PAGE_GAP = 3

# In-text cross-reference pattern: matches "cláusula 12", "Cláusula 4.2",
# "conforme cláusula 10.2" and captures the bare numbering token ("12", "4.2",
# "10.2"), resolved against a candidate clause's ``path`` last segment. The same
# expression ``scripts/find_candidate_clauses.py`` ([M2-08]) uses for golden-set
# curation -- that issue's docstring names this step as the production
# formalisation of it. Kept here (not only in the module that compiles it) so a
# change to what counts as a cross-reference edge moves the fingerprint.
CROSS_REFERENCE_PATTERN = r"cl[áa]usulas?\s+(\d+(?:\.\d+)*)"


def config_fingerprint() -> str:
    """Digest of every value that determines the co-retrieved top-k.

    Threaded into the eval run's
    [infrastructure.evaluation.retrieval_run_schema.RetrievalRunConfig] so a
    report read in isolation says exactly what produced its numbers. Reads the
    module constants at call time (not import time) so a test can monkeypatch one
    and watch the digest move -- mirrors the sibling ``config_fingerprint``s in
    ``reranker_config`` / ``hybrid_config`` / ``embedding_config`` /
    ``lexical_config``.
    """
    payload = "\x00".join(
        (
            str(RESERVED_EXCLUSION_SLOTS),
            str(ADJACENT_SECTION_MAX_PAGE_GAP),
            CROSS_REFERENCE_PATTERN,
        )
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]
