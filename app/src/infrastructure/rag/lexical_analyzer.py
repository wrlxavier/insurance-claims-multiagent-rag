"""Portuguese text analysis for the [M3-03] BM25 lexical retriever.

One analysis path, used identically on the index side (chunk text) and
[M3-04]'s query side, so the tokens a chunk is indexed with and the tokens a
query is scored against cannot drift apart -- the lexical analog of
[infrastructure.rag.embedding_config.format_passage] / ``format_query``.

Pipeline: NFKC-normalise -> lowercase -> Unicode word-split -> per token, either
keep it verbatim (accent-folded) if it is in the committed stemming-exception
list, or Snowball-stem it **and then** accent-fold. The stem-before-fold order
is load-bearing: snowballstemmer's Portuguese suffix rules are defined over
accented text -- ``informação`` stems to ``inform`` while the pre-folded
``informacao`` stems to the useless ``informaca``.

No stopword removal. A generic Portuguese stoplist drops ``não`` / ``nem`` /
``sem`` / ``salvo`` / ``exceto`` -- exactly the words that flip a coverage
clause into an exclusion, which is the hard case the whole M3 milestone
targets. BM25 term saturation plus the near-zero IDF the ``lucene_plus_one``
variant assigns ubiquitous terms already neutralise function words. See
docs/LEXICAL_RETRIEVAL.md.
"""

import re
import unicodedata
from pathlib import Path
from typing import Protocol

from infrastructure.rag.lexical_config import (
    LEXICAL_STEMMING_EXCEPTIONS_PATH,
    STEMMER_LANGUAGE,
)

_TOKEN_RE = re.compile(r"\w+")


class _Stemmer(Protocol):
    """The slice of snowballstemmer's stemmer this module uses."""

    def stemWord(self, word: str) -> str:  # noqa: N802 -- third-party method name
        """Return the stem of a single word."""
        ...


def normalize(text: str) -> str:
    """NFKC-normalise and lowercase. Accents are kept for the stemmer."""
    return unicodedata.normalize("NFKC", text).lower()


def tokenize(text: str) -> list[str]:
    """Split normalised text into Unicode word tokens (accents kept)."""
    return _TOKEN_RE.findall(text)


def fold_accents(token: str) -> str:
    """Strip diacritics (NFKD then drop non-ASCII), matching ``normalize_heading``."""
    return (
        unicodedata.normalize("NFKD", token).encode("ascii", "ignore").decode("utf-8")
    )


def normalized_tokens(text: str) -> list[str]:
    """Run normalize + tokenize + accent-fold, without stemming.

    The vocabulary the stemming-exception list is matched in: its loader uses
    this to assert a listed term is exactly one token.
    """
    return [fold_accents(token) for token in tokenize(normalize(text))]


class TextAnalyzer:
    """Turns raw text into BM25 terms; the one path both retriever sides use."""

    def __init__(self, *, protected_tokens: frozenset[str], stemmer: _Stemmer) -> None:
        """Build over a fixed protected-token set and a Portuguese stemmer."""
        self._protected = protected_tokens
        self._stemmer = stemmer
        # Stemming the same token millions of times across ~4.5k chunks is the
        # index-build cost; the corpus has far fewer distinct tokens than total.
        self._stem_cache: dict[str, str] = {}

    def _stem_then_fold(self, token: str) -> str:
        cached = self._stem_cache.get(token)
        if cached is None:
            cached = fold_accents(self._stemmer.stemWord(token))
            self._stem_cache[token] = cached
        return cached

    def analyze(self, text: str) -> list[str]:
        """Run normalize -> tokenize -> protect-or-stem-then-fold, in that order.

        Order and duplicates are preserved: BM25 scores on term frequency.
        """
        out: list[str] = []
        for token in tokenize(normalize(text)):
            folded = fold_accents(token)
            if folded in self._protected:
                out.append(folded)
            else:
                out.append(self._stem_then_fold(token))
        return out


def build_analyzer(
    exceptions_path: Path = LEXICAL_STEMMING_EXCEPTIONS_PATH,
) -> TextAnalyzer:
    """Load the committed exception list and wire the Snowball Portuguese stemmer."""
    import snowballstemmer

    from infrastructure.rag.lexical_stemming_exceptions import load_stemming_exceptions

    return TextAnalyzer(
        protected_tokens=load_stemming_exceptions(exceptions_path),
        stemmer=snowballstemmer.stemmer(STEMMER_LANGUAGE),
    )
