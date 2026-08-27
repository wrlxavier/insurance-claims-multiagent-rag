"""The pinned embedding-model contract -- [M3-02].

Single source of truth for the embedding model, its revision, the distance
metric, the normalisation decision, and any query/passage prefix. The
embedding pipeline (index side) and [M3-04]'s retriever (query side) both
import this module, so the string a chunk is embedded with and the string a
query is embedded with cannot drift apart. Recorded here, as code, rather
than in a comment -- [M3-04] reads it.

Why these are module constants and not `.env` knobs: the model id, revision,
dimensionality, distance metric, normalisation and prefix together determine
the exact vectors behind a published Recall@k number. Per [M1-09]'s
per-constant rule they are part of the experimental design -- like
``SEED``/``SAMPLE_SIZE`` in ``scripts/sample_parsing_quality.py`` -- and
making them environment-dependent would let a `.env` edit silently change a
published result. ``EMBEDDING_MODEL`` stays in `.env` as the human-facing
name and is cross-checked against ``EMBEDDING_MODEL_ID`` by a test.

Rationale and evidence: docs/EMBEDDINGS.md.
"""

from enum import StrEnum

# Alibaba-NLP/gte-multilingual-base (mGTE, EMNLP 2024 industry track). Chosen
# for evidenced Brazilian-Portuguese retrieval quality (MTEB-PT retrieval
# nDCG@10 76.9; Portuguese is an explicit target language of the mGTE
# training/eval, not incidental). See docs/EMBEDDINGS.md.
EMBEDDING_MODEL_ID = "Alibaba-NLP/gte-multilingual-base"

# Pinned by Hub commit -- NOT the floating ``main`` alias -- so a provider-side
# update cannot silently change the vectors behind an already-published
# number. This revision also pins the ``trust_remote_code`` modelling files
# (``scripts/gte_embedding.py`` in the repo), which are fetched at the same
# revision. Commit last modified 2025-07-05; re-confirm before bumping.
EMBEDDING_MODEL_REVISION = "9bbca17d9273fd0d03d5725c7a4b0f6b45142062"

# Model card: 768-dim output, 8192-token context window.
EMBEDDING_DIMENSIONS = 768
EMBEDDING_MAX_INPUT_TOKENS = 8192

# gte-multilingual-base ships a custom ``NewModel`` architecture, so
# sentence-transformers / transformers must load it with
# ``trust_remote_code=True``. Safe here because the code is fetched at the
# pinned revision above, not from a moving branch. Consumed by the
# embedding-pipeline PR, recorded now alongside the rest of the contract.
EMBEDDING_TRUST_REMOTE_CODE = True


class DistanceMetric(StrEnum):
    """The vector distance metric, fixed once for index and query side alike."""

    COSINE = "cosine"


# Fixed once. Used identically on both sides:
#   - index: the ``halfvec(768)`` column's ANN index takes ``halfvec_cosine_ops``
#     and queries order by the ``<=>`` (cosine distance) operator
#     ([M3-02] embedding-pipeline PR).
#   - query: [M3-04]'s retriever orders by the same ``<=>``.
DISTANCE_METRIC = DistanceMetric.COSINE

# gte-multilingual-base's own usage examples L2-normalise before comparing
# (``F.normalize(embeddings, p=2, dim=1)`` / ``normalize_embeddings=True``).
# With unit vectors, cosine distance and (1 - inner product) coincide. Both
# stored chunk vectors and query vectors are normalised.
NORMALIZE_EMBEDDINGS = True

# gte-multilingual-base takes NO instruction/prefix on either side: its model
# card embeds queries and documents directly, unlike the E5 family whose card
# mandates ``query:`` / ``passage:``. Kept as explicit constants (not absent)
# so a later swap to a prefix model has exactly one place to set them -- and
# ``format_query`` / ``format_passage`` below are the only formatting path
# either side may use.
QUERY_PREFIX = ""
PASSAGE_PREFIX = ""


def format_query(text: str) -> str:
    """Render a query string exactly as it must be embedded (query side, [M3-04])."""
    return f"{QUERY_PREFIX}{text}"


def format_passage(text: str) -> str:
    """Render a chunk's text exactly as it must be embedded (index side)."""
    return f"{PASSAGE_PREFIX}{text}"
