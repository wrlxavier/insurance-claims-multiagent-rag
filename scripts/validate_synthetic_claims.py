#!/usr/bin/env python3
"""Validate the synthetic claim set against its schema and the parsed corpus [M2-04].

Mirrors ``scripts/validate_golden_set.py`` exactly: loads
``data/synthetic_claims/claims.jsonl``, validates each row against
[infrastructure.evaluation.synthetic_claims_schema.SyntheticClaim] (schema,
enum vocabulary, and the missing-fact-consistency rule all fail loudly via
Pydantic), and checks two structural properties a schema alone can't
express:

- **document_id exists**: every row's ``document_id`` must be a real row in
  ``data/policies/manifest.csv``.
- **reference_clause_ids exist**: every id must be a real ``clause_id`` in
  the built corpus (``build/parsed_clauses.jsonl``).

Also asserts ``claim_id`` is unique. DoD-floor counts (>=30 total, >=10
insufficient_information, spread across product lines) are reported for
visibility but not enforced here -- the same split
``scripts/validate_golden_set.py`` keeps from its own draft script's
``print_coverage_report``: this script's job is structural validity, not
curation-floor bookkeeping.

Run via ``make validate-synthetic-claims``.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from infrastructure.evaluation.synthetic_claims_schema import SyntheticClaim
from infrastructure.parsing.corpus_artifact import JSONL_PATH, read_parsed_clauses_jsonl
from infrastructure.parsing.manifest import read_manifest

CLAIMS_PATH = Path("data/synthetic_claims/claims.jsonl")
MANIFEST_PATH = Path("data/policies/manifest.csv")


class SyntheticClaimsValidationError(Exception):
    """Raised when the synthetic claim set fails a structural or schema check."""


def load_valid_clause_ids() -> set[str]:
    """Return the set of every clause_id in the built corpus."""
    if not JSONL_PATH.exists():
        raise FileNotFoundError(
            f"{JSONL_PATH} does not exist. Run `make fetch-corpus-artifacts` "
            "(pre-built corpus) or `make parse` (full rebuild) first."
        )
    return {record.clause_id for record in read_parsed_clauses_jsonl(JSONL_PATH)}


def load_document_product_lines(manifest_path: Path) -> dict[str, str]:
    """Return {document_id: product_line} from manifest.csv."""
    return {row["id"]: row["product_line"] for row in read_manifest(manifest_path)}


def load_claims(path: Path) -> list[SyntheticClaim]:
    """Parse every line of the synthetic claims JSONL file as a SyntheticClaim."""
    claims = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                claims.append(SyntheticClaim.model_validate(json.loads(line)))
            except (json.JSONDecodeError, ValidationError) as error:
                raise SyntheticClaimsValidationError(
                    f"{path}:{line_number}: invalid synthetic claim -- {error}"
                ) from error
    return claims


def check_document_ids_exist(
    claims: list[SyntheticClaim], valid_document_ids: set[str]
) -> None:
    """Assert every claim's document_id exists in manifest.csv."""
    for claim in claims:
        if claim.document_id not in valid_document_ids:
            raise SyntheticClaimsValidationError(
                f"claim {claim.claim_id!r} references unknown document_id "
                f"{claim.document_id!r}"
            )


def check_reference_clause_ids_exist(
    claims: list[SyntheticClaim], valid_clause_ids: set[str]
) -> None:
    """Assert every claim's reference_clause_ids all exist in the corpus."""
    for claim in claims:
        unknown = [
            clause_id
            for clause_id in claim.reference_clause_ids
            if clause_id not in valid_clause_ids
        ]
        if unknown:
            raise SyntheticClaimsValidationError(
                f"claim {claim.claim_id!r} references unknown clause_id(s) {unknown!r}"
            )


def check_claim_ids_unique(claims: list[SyntheticClaim]) -> None:
    """Assert claim_id is unique across the whole set."""
    seen: set[str] = set()
    for claim in claims:
        if claim.claim_id in seen:
            raise SyntheticClaimsValidationError(
                f"duplicate claim_id {claim.claim_id!r}"
            )
        seen.add(claim.claim_id)


def main() -> None:
    """Validate data/synthetic_claims/claims.jsonl and print a summary."""
    if not CLAIMS_PATH.exists():
        raise FileNotFoundError(
            f"{CLAIMS_PATH} does not exist. Run `make finalize-synthetic-claims` first."
        )

    valid_clause_ids = load_valid_clause_ids()
    document_product_lines = load_document_product_lines(MANIFEST_PATH)

    claims = load_claims(CLAIMS_PATH)
    check_document_ids_exist(claims, set(document_product_lines))
    check_reference_clause_ids_exist(claims, valid_clause_ids)
    check_claim_ids_unique(claims)

    by_verdict: dict[str, int] = {}
    by_product_line: dict[str, int] = {}
    for claim in claims:
        by_verdict[claim.expected_verdict.value] = (
            by_verdict.get(claim.expected_verdict.value, 0) + 1
        )
        line = document_product_lines.get(claim.document_id, "?")
        by_product_line[line] = by_product_line.get(line, 0) + 1

    print(f"{CLAIMS_PATH}: {len(claims)} claim(s)")
    print("By expected_verdict:")
    for verdict, count in sorted(by_verdict.items()):
        print(f"  {verdict}: {count}")
    print("By product_line:")
    for line, count in sorted(by_product_line.items()):
        print(f"  {line}: {count}")

    print(f"\nTotal: {len(claims)} claim(s). OK.")


if __name__ == "__main__":
    main()
