#!/usr/bin/env python3
"""PII safety net for synthetic claim narratives [M2-04].

A pure-regex backstop, run over every LLM-drafted narrative before it is
written to the review CSV. The primary defense is the prompt itself (see
``scripts/draft_synthetic_claims.py``'s ``_SHARED_RULES``); this module exists
because a prompt instruction is not a guarantee. It only ever **flags** a
narrative for extra human scrutiny -- it never drops or silently regenerates a
row, since the reviewing author (not this script) makes the actual call.

Known limits, stated rather than hidden:

- The plate and CPF patterns are precise (fixed digit/letter shapes) and
  should have very few false positives, but also cannot catch a plate or CPF
  written out in prose ("zero zero um...").
- The proper-name heuristic (two consecutive capitalized words) is
  deliberately loose and WILL false-positive on legitimate multi-word insurer
  names, city names, and Portuguese phrases that happen to capitalize two
  words in a row. It is a prompt for human attention, not a verdict.
"""

from __future__ import annotations

import re

BR_PLATE_OLD = re.compile(r"\b[A-Za-z]{3}-?\d{4}\b")
BR_PLATE_MERCOSUL = re.compile(r"\b[A-Za-z]{3}\d[A-Za-z]\d{2}\b")
CPF_LIKE = re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b")
RG_LIKE = re.compile(r"\b\d{1,2}\.?\d{3}\.?\d{3}-?[0-9Xx]\b")

# Two consecutive capitalized tokens not immediately following sentence-ending
# punctuation -- a loose proxy for "looks like a person's full name". Known to
# over-fire on insurer/brand/city names; see the module docstring.
PROPER_NAME_HEURISTIC = re.compile(
    r"(?<![.!?]\s)\b[A-ZÀ-Ý][a-zà-ÿ]+\s+[A-ZÀ-Ý][a-zà-ÿ]+\b"
)

_CHECKS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("plate_old_format", BR_PLATE_OLD),
    ("plate_mercosul_format", BR_PLATE_MERCOSUL),
    ("cpf_like", CPF_LIKE),
    ("rg_like", RG_LIKE),
    ("possible_proper_name", PROPER_NAME_HEURISTIC),
)


def scan_narrative_for_pii(text: str) -> list[str]:
    """Return the hit-type labels found in `text`; empty means clean.

    Flags only -- never used to auto-reject or auto-regenerate a narrative.
    """
    return [label for label, pattern in _CHECKS if pattern.search(text)]
