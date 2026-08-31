"""The pinned cross-encoder-reranker contract -- [M3-05].

Single source of truth for the reranker model, its revision, its input-length
cap and how deep the fused candidate list is re-scored. The eval harness (query
side) and M4's graph node both reach the reranker through this module, so the
model a golden-set number was measured under and the model M4 runs cannot drift
apart. Recorded here, as code, not in a comment.

Why these are module constants and not `.env` knobs: the model id, revision,
input cap and candidate depth together determine the exact ranking behind a
published MRR / Recall@10 number. Per [M1-09]'s per-constant rule they are
experimental design -- like ``SEED`` / ``SAMPLE_SIZE`` in
``scripts/sample_parsing_quality.py`` -- and making them environment-dependent
would let a `.env` edit silently move a published result. ``RERANKER_MODEL``
stays in `.env` as the human-facing name and is cross-checked against
``RERANKER_MODEL_ID`` by a test, exactly as ``EMBEDDING_MODEL`` is against
``EMBEDDING_MODEL_ID``. This issue introduces **no** new `.env` key -- the
analog of ``docs/LEXICAL_RETRIEVAL.md``'s "zero" and
``docs/HYBRID_RETRIEVAL.md``'s "zero".

Rationale and evidence: docs/RERANKING.md.
"""

import hashlib

# Alibaba-NLP/gte-multilingual-reranker-base -- the reranking half of the same
# mGTE work as the pinned embedder (Alibaba-NLP/gte-multilingual-base). The
# paper docs/EMBEDDINGS.md already cites ("...Text Representation *and Reranking*
# Models for Multilingual Text", arXiv 2407.19669) is this model's paper;
# Portuguese is an explicit target language, not incidental. Same 8192-token
# window, same architecture family. See docs/RERANKING.md.
RERANKER_MODEL_ID = "Alibaba-NLP/gte-multilingual-reranker-base"

# Pinned by Hub commit -- NOT the floating ``main`` alias -- so a provider-side
# re-upload cannot silently change the weights behind an already-published
# number. Commit last modified 2025-07-05; re-confirm before bumping.
#
# Same caveat as the embedder (embedding_config.py): the ``trust_remote_code``
# modelling code is loaded from ``Alibaba-NLP/new-impl`` at *its* ``main``, not
# this revision -- HF's ``repo--module`` auto_map form does not pin it. The
# ``embed`` group's ``transformers`` pin (the 4.4x line) is what stabilises that
# RoPE code path; the reranker's config.json declares the same
# ``transformers_version 4.39.1`` as the embedder's.
RERANKER_MODEL_REVISION = "8215cf04918ba6f7b6a62bb44238ce2953d8831c"

# Model card / config.json: 8192-token context window. Passed to the
# cross-encoder as ``max_length`` so a long clause is truncated deterministically
# rather than by the tokenizer's silent default.
RERANKER_MAX_INPUT_TOKENS = 8192

# gte-multilingual-reranker-base ships the same custom ``New*`` architecture as
# the embedder (auto_map -> ``Alibaba-NLP/new-impl``), so it must be loaded with
# ``trust_remote_code=True``. Safe because the weights are fetched at the pinned
# revision above. Security-sensitive -- a code constant, never `.env`-flippable,
# per the ``EMBEDDING_TRUST_REMOTE_CODE`` precedent.
RERANKER_TRUST_REMOTE_CODE = True

# How many of the fused hybrid candidates the cross-encoder re-scores before the
# final top-k cut. Set to 10 -- the shallowest point -- from the ``make
# tune-reranking`` curve in docs/RERANKING.md: reranking the hybrid top-10 lifts
# R@1 / R@5 / MRR / nDCG with R@10 unchanged and exclusion-clause recall held at
# 92.6%, while every deeper depth trades that top-rank gain away and depth >= 30
# pushes exclusion clauses out of the kept context (the M3-05 DoD's named
# regression) and drops R@10 below the no-rerank baseline. A code constant: it
# changes the reranked ranking the golden-set numbers are measured on.
RERANK_CANDIDATE_DEPTH = 10


def config_fingerprint() -> str:
    """Digest of every contract value that determines the reranked order.

    [infrastructure.rag.reranker_cache.CachingReranker] prepends this to its
    per-(query, passage) key, so a cached score can never be served under a
    different model id, revision, input cap or candidate depth than the one it
    was computed under. Threaded into the eval run's
    [infrastructure.evaluation.retrieval_run_schema.RetrievalRunConfig] so a
    report read in isolation says exactly what produced its numbers.

    Reads the module constants at call time (not import time) so a test can
    monkeypatch one and see the fingerprint move -- mirrors the sibling
    ``config_fingerprint``s in ``embedding_config`` / ``lexical_config`` /
    ``hybrid_config``.
    """
    payload = "\x00".join(
        (
            RERANKER_MODEL_ID,
            RERANKER_MODEL_REVISION,
            str(RERANKER_MAX_INPUT_TOKENS),
            str(RERANK_CANDIDATE_DEPTH),
        )
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]
