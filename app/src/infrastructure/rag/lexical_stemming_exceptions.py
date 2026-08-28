"""Loader for the committed stemming-exception list -- [M3-03].

Mirrors [infrastructure.parsing.rules_loader]: a headered CSV under ``data/``,
read with the stdlib ``csv`` module. Each ``term`` is a domain word the
Portuguese stemmer damages (over-stems into a common word, or mangles), kept
verbatim in the BM25 vocabulary instead. Committed as data per the M3-03 DoD;
seeded only with the SUSEP identifiers and grown from measured misses -- the
rationale for every entry is in the ``note`` column and docs/LEXICAL_RETRIEVAL.md.
"""

import csv
from pathlib import Path

from infrastructure.rag.lexical_analyzer import normalized_tokens


def load_stemming_exceptions(csv_path: Path) -> frozenset[str]:
    """Load the exception list as the set of protected (normalised) tokens.

    CSV header: ``term,note``. ``term`` is normalised (NFKC, lowercase,
    accent-fold) exactly as [infrastructure.rag.lexical_analyzer] normalises
    corpus tokens, and must be a single token -- BM25 is bag-of-words, so a
    multi-word phrase cannot be protected as a unit and a row that is not one
    token is a data error. ``note`` is human rationale, ignored here.
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} does not exist. It is committed at "
            "data/rag/lexical_stemming_exceptions.csv."
        )

    protected: set[str] = set()
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            raw = row["term"].strip()
            tokens = normalized_tokens(raw)
            if len(tokens) != 1:
                raise ValueError(
                    f"stemming-exception term {raw!r} normalises to {tokens!r}; "
                    "each row must be exactly one token."
                )
            protected.add(tokens[0])
    return frozenset(protected)
