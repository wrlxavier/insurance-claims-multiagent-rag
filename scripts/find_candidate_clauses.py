#!/usr/bin/env python3
r"""Deterministic candidate-clause search for golden-set curation [M2-08].

Supports the author's completeness check in [M2-01]'s three-layer authoring
flow (see ``docs/EVALUATION.md``): given a ``clause_id``, print a short list
of *structurally* related clauses in the same document that a human should
look at when deciding whether a golden question's ``reference_clause_ids``
is complete. This is conceptually an early, reduced version of [M3-06]'s
exclusion co-retrieval, brought forward as a curation tool before it exists
as part of the product.

Purely structural/deterministic -- no LLM call, no content judgment. Three
signals, computed against clauses in the same document, merged per candidate
when more than one applies:

- shared ``parent_id`` with the query clause (siblings in the M1 clause tree)
- matching ``bundle_section``
- a textual cross-reference ("cláusula N[.M...]") in the query clause's body
  that points at the candidate's numbering (its ``path``'s last segment)

Candidates are sorted by number of matching signals (most first), then by
``clause_id``, and capped at ``--max-candidates`` (default 10).

IMPORTANT: an empty candidate list does NOT mean no related clause exists.
This script only surfaces structural signals already present in the M1
clause tree -- it never decides relevance and cannot see relationships it
has no structural signal for. Checking for that false negative remains the
author's responsibility as part of [M2-01]'s completeness check, not this
script's.

Reads ``build/parsed_clauses.jsonl`` (run `make fetch-corpus-artifacts` or
`make parse` first if missing). Run directly:

    PYTHONPATH=app/src uv run python scripts/find_candidate_clauses.py \\
        --clause-id "<clause_id>"
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass

from infrastructure.parsing.clause_schema import ParsedClauseRecord
from infrastructure.parsing.corpus_artifact import JSONL_PATH, read_parsed_clauses_jsonl

DEFAULT_MAX_CANDIDATES = 10

_CROSS_REFERENCE_PATTERN = re.compile(r"cl[áa]usulas?\s+(\d+(?:\.\d+)*)", re.IGNORECASE)


class ClauseNotFoundError(Exception):
    """Raised when the requested clause_id isn't in the parsed corpus."""


@dataclass(frozen=True)
class Candidate:
    """One candidate clause, with the structural signal(s) that matched it."""

    clause_id: str
    title: str
    reasons: tuple[str, ...]


def load_corpus() -> list[ParsedClauseRecord]:
    """Load the parsed corpus, failing loudly if `make parse` hasn't run."""
    if not JSONL_PATH.exists():
        raise FileNotFoundError(
            f"{JSONL_PATH} does not exist. Run `make fetch-corpus-artifacts` "
            "(pre-built corpus) or `make parse` (full rebuild) first."
        )
    return read_parsed_clauses_jsonl(JSONL_PATH)


def find_clause(
    records: list[ParsedClauseRecord], clause_id: str
) -> ParsedClauseRecord:
    """Return the record matching `clause_id`, or raise ClauseNotFoundError."""
    for record in records:
        if record.clause_id == clause_id:
            return record
    raise ClauseNotFoundError(f"clause_id {clause_id!r} not found in {JSONL_PATH}")


def extract_cross_references(text: str) -> set[str]:
    """Return the numbering tokens textually referenced in `text`.

    Reduced-scope version of the cross-reference detection [M3-06] will
    later formalise for production co-retrieval: matches "cláusula 12",
    "Cláusula 4.2", etc. and returns the bare numbering ("12", "4.2").
    """
    return {match.group(1) for match in _CROSS_REFERENCE_PATTERN.finditer(text)}


def find_candidates(
    records: list[ParsedClauseRecord],
    clause_id: str,
    *,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> list[Candidate]:
    """Return up to `max_candidates` structurally-related clauses for `clause_id`.

    See the module docstring for the three signals and their semantics. This
    is a short list to *look at*, not a verdict.
    """
    target = find_clause(records, clause_id)
    referenced_numbers = extract_cross_references(target.text)

    reasons_by_id: dict[str, list[str]] = {}
    titles_by_id: dict[str, str] = {}

    for record in records:
        if (
            record.document_id != target.document_id
            or record.clause_id == target.clause_id
        ):
            continue
        reasons: list[str] = []
        if target.parent_id is not None and record.parent_id == target.parent_id:
            reasons.append("shared_parent")
        if (
            target.bundle_section is not None
            and record.bundle_section == target.bundle_section
        ):
            reasons.append("matching_bundle_section")
        numbering = record.path.rsplit("/", 1)[-1]
        if numbering in referenced_numbers:
            reasons.append(f"cross_reference:{numbering}")
        if reasons:
            reasons_by_id[record.clause_id] = reasons
            titles_by_id[record.clause_id] = record.title

    candidates = [
        Candidate(clause_id=cid, title=titles_by_id[cid], reasons=tuple(reasons))
        for cid, reasons in reasons_by_id.items()
    ]
    candidates.sort(key=lambda c: (-len(c.reasons), c.clause_id))
    return candidates[:max_candidates]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clause-id",
        required=True,
        help="Query clause_id to find candidates for, e.g. '1:.../7'.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=DEFAULT_MAX_CANDIDATES,
        help=f"Cap on candidates returned (default {DEFAULT_MAX_CANDIDATES}).",
    )
    return parser.parse_args()


def main() -> None:
    """Look up candidates for --clause-id and print them."""
    args = _parse_args()
    records = load_corpus()
    candidates = find_candidates(
        records, args.clause_id, max_candidates=args.max_candidates
    )

    if not candidates:
        print(f"No structural candidates found for {args.clause_id!r}.")
        print(
            "This does NOT mean no related clause exists -- checking for that "
            "false negative is the author's responsibility, not this script's."
        )
        return

    print(f"Candidates for {args.clause_id!r} ({len(candidates)}):")
    for candidate in candidates:
        reasons = ", ".join(candidate.reasons)
        print(f"  {candidate.clause_id}  [{reasons}]  {candidate.title}")


if __name__ == "__main__":
    main()
