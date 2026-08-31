"""Hand-rolled Okapi BM25 -- [M3-03].

Pure functions over already-analysed token lists, no I/O and no third-party
dependency -- mirroring how [infrastructure.evaluation.retrieval_metrics]
hand-rolls nDCG rather than pulling an IR library. ``rank-bm25`` / ``bm25s``
would both drag ``numpy`` (``bm25s`` also ``scipy``) into the default
environment, which is deliberately numpy/torch-free.

IDF is the Lucene / BM25+ form ``ln(1 + (N - df + 0.5) / (df + 0.5))``, always
> 0 -- see [infrastructure.rag.lexical_config.IDF_VARIANT] for why the classic
Okapi IDF's negative branch is unacceptable here.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class BM25Index:
    """An inverted index plus per-term IDF, ready to score queries."""

    doc_ids: tuple[str, ...]
    doc_len: tuple[int, ...]
    avg_doc_len: float
    # term -> ((doc_idx, term_frequency), ...)
    postings: dict[str, tuple[tuple[int, int], ...]]
    idf: dict[str, float]
    k1: float
    b: float


def build_bm25_index(
    docs: Sequence[tuple[str, Sequence[str]]], *, k1: float, b: float
) -> BM25Index:
    """Build the index from ``(doc_id, analysed tokens)`` pairs."""
    doc_ids = tuple(doc_id for doc_id, _ in docs)
    doc_len = tuple(len(tokens) for _, tokens in docs)
    n = len(docs)
    avg_doc_len = (sum(doc_len) / n) if n else 0.0

    entries_by_term: dict[str, list[tuple[int, int]]] = {}
    for doc_idx, (_, tokens) in enumerate(docs):
        counts: dict[str, int] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
        for token, term_frequency in counts.items():
            entries_by_term.setdefault(token, []).append((doc_idx, term_frequency))

    postings = {term: tuple(entries) for term, entries in entries_by_term.items()}
    idf = {
        term: math.log(1.0 + (n - len(entries) + 0.5) / (len(entries) + 0.5))
        for term, entries in postings.items()
    }
    return BM25Index(
        doc_ids=doc_ids,
        doc_len=doc_len,
        avg_doc_len=avg_doc_len,
        postings=postings,
        idf=idf,
        k1=k1,
        b=b,
    )


def score(index: BM25Index, query_tokens: Sequence[str]) -> dict[str, float]:
    """BM25 score keyed by ``doc_id``, for docs sharing >=1 term with the query.

    A doc with no query term in common is absent from the result, not listed
    with 0.0. Repeated query terms count once (Okapi ignores query-side term
    frequency).
    """
    if index.avg_doc_len == 0.0:
        return {}
    scores: dict[int, float] = {}
    for token in set(query_tokens):
        entries = index.postings.get(token)
        if entries is None:
            continue
        idf = index.idf[token]
        for doc_idx, term_frequency in entries:
            length_norm = (
                1.0 - index.b + index.b * index.doc_len[doc_idx] / index.avg_doc_len
            )
            contribution = (
                idf
                * (term_frequency * (index.k1 + 1.0))
                / (term_frequency + index.k1 * length_norm)
            )
            scores[doc_idx] = scores.get(doc_idx, 0.0) + contribution
    return {index.doc_ids[doc_idx]: value for doc_idx, value in scores.items()}


def top_n(
    index: BM25Index, query_tokens: Sequence[str], n: int
) -> list[tuple[str, float]]:
    """The ``n`` highest-scoring docs; ties broken by ``doc_id`` ascending.

    The tie-break keeps a committed Recall@k number reproducible run to run.
    """
    ranked = sorted(
        score(index, query_tokens).items(), key=lambda item: (-item[1], item[0])
    )
    return ranked[:n]
