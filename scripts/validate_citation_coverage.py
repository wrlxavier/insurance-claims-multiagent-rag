#!/usr/bin/env python3
"""Enforce end-to-end citation coverage against the committed snapshot [M4-10].

The [M4-10] DoD asks for an **automated** check that "100% of assertions carry
a clause id, and every id exists in the corpus. CI fails otherwise" -- by test,
not by review. This script is that check, and it is deliberately a script
rather than only a pytest case: CI runs ``pytest -m unit`` *before* it fetches
the corpus artifacts, so the corpus half of the property has to run as its own
step, after ``make fetch-corpus-artifacts``, exactly like
``make validate-golden-set`` already does.

It replays ``eval/end_to_end_citations.json`` -- the committed per-claim
snapshot ``make eval-end-to-end`` writes -- so it needs no database, no GPU, no
``embed`` group and no LLM key. Three properties, all structural:

1. **Every assertion on a settled verdict carries at least one clause id.** An
   ``insufficient_information`` assessment carries free prose and no ids by
   design (``nodes/compatibility._abstain``), so those rows are excluded --
   including them would fail the check for behaving correctly.
2. **Every assertion clause id exists in the corpus** (``build/
   parsed_clauses.jsonl``), so a re-parse that renames a ``clause_id`` breaks
   visibly rather than silently invalidating a published citation.
3. **Every recommendation citation exists in the corpus and was retrieved for
   that claim** -- the [M4-08] "never introduce a citation no upstream node
   produced" guarantee, re-checked here on a real run rather than on a fake.

Missing input is an error, never a skip: a check that quietly passes when its
evidence is absent is not a gate. Run via ``make validate-citation-coverage``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scripts.validate_synthetic_claims import load_valid_clause_ids

SNAPSHOT_PATH = Path("eval/end_to_end_citations.json")

# The verdicts for which an assertion must be grounded. An abstention is the
# one case where citation-free prose is the correct output.
SETTLED_VERDICTS = frozenset({"compatible", "incompatible"})


class CitationCoverageError(Exception):
    """Raised when a run's citations fail a structural coverage check."""


def load_snapshot(path: Path = SNAPSHOT_PATH) -> dict[str, Any]:
    """Load the committed snapshot, failing loudly when it is absent."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Run `make eval-end-to-end` (which writes it) "
            "and commit the result."
        )
    loaded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if not loaded.get("claims"):
        raise CitationCoverageError(f"{path} contains no claims")
    return loaded


def check_every_assertion_carries_a_clause_id(claims: list[dict[str, Any]]) -> int:
    """Assert every assertion on a settled verdict cites at least one clause.

    Returns the number of assertions checked, so the caller can report the
    denominator rather than an unqualified "100%".
    """
    checked = 0
    for claim in claims:
        if claim.get("compatibility_verdict") not in SETTLED_VERDICTS:
            continue
        assertions = claim.get("assertions") or []
        if not assertions:
            raise CitationCoverageError(
                f"claim {claim['claim_id']!r} has a settled verdict "
                f"{claim['compatibility_verdict']!r} but no grounded assertions"
            )
        for index, assertion in enumerate(assertions, start=1):
            checked += 1
            if not assertion.get("clause_ids"):
                raise CitationCoverageError(
                    f"claim {claim['claim_id']!r} assertion {index} carries no "
                    "clause id"
                )
    return checked


def check_assertion_clause_ids_exist(
    claims: list[dict[str, Any]], valid_clause_ids: set[str]
) -> None:
    """Assert every cited clause id is a real clause_id in the built corpus."""
    for claim in claims:
        for index, assertion in enumerate(claim.get("assertions") or [], start=1):
            unknown = [
                clause_id
                for clause_id in assertion.get("clause_ids") or []
                if clause_id not in valid_clause_ids
            ]
            if unknown:
                raise CitationCoverageError(
                    f"claim {claim['claim_id']!r} assertion {index} cites clause "
                    f"id(s) absent from the corpus: {unknown!r}"
                )


def check_recommendation_citations(
    claims: list[dict[str, Any]], valid_clause_ids: set[str]
) -> None:
    """Assert recommendation citations exist and were retrieved for that claim."""
    for claim in claims:
        cited = list(claim.get("recommendation_citation_ids") or [])
        retrieved = set(claim.get("retrieved_clause_ids") or [])
        unknown = [c for c in cited if c not in valid_clause_ids]
        if unknown:
            raise CitationCoverageError(
                f"claim {claim['claim_id']!r} recommends clause id(s) absent from "
                f"the corpus: {unknown!r}"
            )
        ungrounded = [c for c in cited if c not in retrieved]
        if ungrounded:
            raise CitationCoverageError(
                f"claim {claim['claim_id']!r} recommends clause id(s) retrieval "
                f"never returned: {ungrounded!r}"
            )


def main() -> None:
    """Validate eval/end_to_end_citations.json and print a summary."""
    snapshot = load_snapshot()
    claims = list(snapshot["claims"])
    valid_clause_ids = load_valid_clause_ids()

    n_assertions = check_every_assertion_carries_a_clause_id(claims)
    check_assertion_clause_ids_exist(claims, valid_clause_ids)
    check_recommendation_citations(claims, valid_clause_ids)

    settled = [c for c in claims if c.get("compatibility_verdict") in SETTLED_VERDICTS]
    n_citations = sum(len(c.get("recommendation_citation_ids") or []) for c in claims)
    provenance = snapshot.get("provenance", {})
    print(f"{SNAPSHOT_PATH}: {len(claims)} claim(s)")
    print(f"  generated: {provenance.get('generated_at_utc', '?')}")
    print(f"  settled verdicts: {len(settled)}")
    print(f"  assertions with a clause id: {n_assertions}/{n_assertions} (100%)")
    print(f"  recommendation citations checked: {n_citations}")
    print(f"  corpus clause ids: {len(valid_clause_ids)}")
    print("Citation coverage OK.")


if __name__ == "__main__":
    main()
