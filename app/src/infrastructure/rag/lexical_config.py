"""The pinned lexical-retrieval contract -- [M3-03].

Single source of truth for the BM25 lexical retriever: the analyzer version,
the stemmer language, the Okapi BM25 parameters, the IDF variant, the indexed
text field, and the on-disk stemming-exception list. The index side (chunk
text) and [M3-04]'s query side both go through the one analyzer built from this
module, so the tokens a chunk is indexed with and the tokens a query is scored
with cannot drift apart.

Why these are module constants and not ``.env`` knobs: the analyzer version,
BM25 ``k1``/``b``, the IDF variant and the indexed field together determine the
exact Recall@k number published in ``docs/LEXICAL_RETRIEVAL.md``. Per [M1-09]'s
per-constant rule they are experimental design -- like ``SEED`` / ``SAMPLE_SIZE``
in ``scripts/sample_parsing_quality.py`` -- and making them environment-dependent
would let a ``.env`` edit silently move a published result. This issue introduces
**no** ``.env`` key (the analog of docs/EMBEDDINGS.md's "exactly one constant
moved" -- here, zero).

Rationale and evidence: docs/LEXICAL_RETRIEVAL.md.
"""

import hashlib
from pathlib import Path
from typing import Literal

# Bump on ANY change that moves the tokens the analyzer emits: the pipeline, the
# stemmer, the tokeniser regex, OR the contents of the stemming-exception CSV.
# It is the documented signal that the numbers in docs/LEXICAL_RETRIEVAL.md are
# stale and the eval must be re-run.
LEXICAL_ANALYZER_VERSION = "v1"

# snowballstemmer's "portuguese" algorithm -- the same Snowball family Postgres'
# built-in ``portuguese`` text-search config uses, so a future DB-side lexical
# path ([M3-04]) stays consistent with this one. Rejected alternatives (nltk
# RSLP, PyStemmer) and why: docs/LEXICAL_RETRIEVAL.md.
STEMMER_LANGUAGE = "portuguese"

# Okapi BM25. Standard defaults; tuning the curve is [M3-05]/[M3-08]'s, not this
# issue's. They change the published Recall@k, so they stay code constants.
BM25_K1 = 1.5
BM25_B = 0.75

# Lucene / BM25+ IDF: ``ln(1 + (N - df + 0.5) / (df + 0.5))`` -- always > 0.
# Classic Okapi IDF goes negative for any term in more than half the documents;
# with the deliberate no-stopword decision (docs/LEXICAL_RETRIEVAL.md) ubiquitous
# Portuguese function words would then get negative weight and *penalise* the
# true clause under a verbose analyst question. This form asymptotes to ~0 for
# ubiquitous terms instead -- harmless, not harmful.
IDF_VARIANT = "lucene_plus_one"

# Which ``ChunkRecord`` field BM25 indexes. ``text`` (the embedding side's
# string, ancestor-path breadcrumb prepended) rather than ``display_text``:
# parity with the dense side for [M3-04]'s rank fusion, and the DoD's exact
# terms often live only in an ancestor heading that ``display_text`` strips.
# A constant so [M3-08] can A/B ``display_text`` in one line.
LEXICAL_INDEX_TEXT_FIELD: Literal["text", "display_text"] = "text"

# Domain terms the stemmer damages, kept verbatim (unstemmed). Committed as data
# per the DoD; seeded only with the SUSEP identifiers (codes, not words -- per
# this repo's own convention) and grown from measured misses, never assumed.
LEXICAL_STEMMING_EXCEPTIONS_PATH = Path("data/rag/lexical_stemming_exceptions.csv")


def config_fingerprint(*, exception_tokens: frozenset[str]) -> str:
    """Digest of every value that determines the tokens this retriever scores on.

    Threaded into [infrastructure.evaluation.retrieval_run_schema.
    RetrievalRunConfig] so a report read in isolation says exactly what produced
    its numbers, and reserved for [M3-04]'s query-side cache key. Reads the
    module constants at call time (not import time) so a test can monkeypatch one
    and watch the digest move, mirroring [infrastructure.rag.embedding_config.
    config_fingerprint].
    """
    exceptions_digest = hashlib.sha256(
        "\x00".join(sorted(exception_tokens)).encode()
    ).hexdigest()
    payload = "\x00".join(
        (
            LEXICAL_ANALYZER_VERSION,
            STEMMER_LANGUAGE,
            str(BM25_K1),
            str(BM25_B),
            IDF_VARIANT,
            LEXICAL_INDEX_TEXT_FIELD,
            exceptions_digest,
        )
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]
