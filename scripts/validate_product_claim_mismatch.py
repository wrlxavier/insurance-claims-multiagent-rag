#!/usr/bin/env python3
"""Validate the product/claim mismatch set against its schema and the corpus [M2-05].

Mirrors ``scripts/validate_synthetic_claims.py`` against
``data/synthetic_claims/product_claim_mismatch.jsonl``, plus two dataset-
specific invariants a generic schema can't express -- both are the entire
point of this dataset, so a validator that doesn't check them isn't really
validating it:

- **non-CASCO only**: every row's ``document_id`` must resolve to a
  product_line other than ``CASCO`` in ``data/policies/manifest.csv`` -- a
  mismatch claim targets a document that structurally cannot cover
  own-vehicle damage, and CASCO is the one line that can.
- **incompatible only**: every row's ``expected_verdict`` must be
  ``incompatible`` -- there is no other verdict a product/claim mismatch
  claim can carry by construction.

Also checks, as ``validate_synthetic_claims.py`` does: schema conformance
(via [infrastructure.evaluation.synthetic_claims_schema.SyntheticClaim]),
``document_id`` and ``reference_clause_ids`` existence against the real
corpus, and ``claim_id`` uniqueness.

Run via ``make validate-product-claim-mismatch``.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from infrastructure.evaluation.golden_set_schema import ExpectedVerdict
from infrastructure.evaluation.synthetic_claims_schema import SyntheticClaim
from infrastructure.parsing.corpus_artifact import JSONL_PATH, read_parsed_clauses_jsonl
from infrastructure.parsing.manifest import read_manifest

MISMATCH_CLAIMS_PATH = Path("data/synthetic_claims/product_claim_mismatch.jsonl")
MANIFEST_PATH = Path("data/policies/manifest.csv")

NON_CASCO_PRODUCT_LINES = frozenset({"RCF-A", "ASSIST", "GAR.EST", "CARTA VERDE"})


class ProductClaimMismatchValidationError(Exception):
    """Raised when the mismatch set fails a structural, schema, or scope check."""


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
    """Parse every line of the mismatch JSONL file as a SyntheticClaim."""
    claims = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                claims.append(SyntheticClaim.model_validate(json.loads(line)))
            except (json.JSONDecodeError, ValidationError) as error:
                raise ProductClaimMismatchValidationError(
                    f"{path}:{line_number}: invalid claim -- {error}"
                ) from error
    return claims


def check_document_ids_exist(
    claims: list[SyntheticClaim], valid_document_ids: set[str]
) -> None:
    """Assert every claim's document_id exists in manifest.csv."""
    for claim in claims:
        if claim.document_id not in valid_document_ids:
            raise ProductClaimMismatchValidationError(
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
            raise ProductClaimMismatchValidationError(
                f"claim {claim.claim_id!r} references unknown clause_id(s) {unknown!r}"
            )


def check_claim_ids_unique(claims: list[SyntheticClaim]) -> None:
    """Assert claim_id is unique across the whole set."""
    seen: set[str] = set()
    for claim in claims:
        if claim.claim_id in seen:
            raise ProductClaimMismatchValidationError(
                f"duplicate claim_id {claim.claim_id!r}"
            )
        seen.add(claim.claim_id)


def check_targets_non_casco_documents(
    claims: list[SyntheticClaim], document_product_lines: dict[str, str]
) -> None:
    """Assert every claim targets a document outside the CASCO product line."""
    for claim in claims:
        line = document_product_lines.get(claim.document_id)
        if line not in NON_CASCO_PRODUCT_LINES:
            raise ProductClaimMismatchValidationError(
                f"claim {claim.claim_id!r} targets document_id "
                f"{claim.document_id!r} (product_line={line!r}), but every "
                "product/claim mismatch claim must target a non-CASCO document"
            )


def check_verdict_is_always_incompatible(claims: list[SyntheticClaim]) -> None:
    """Assert every claim's expected_verdict is `incompatible`."""
    for claim in claims:
        if claim.expected_verdict != ExpectedVerdict.INCOMPATIBLE:
            raise ProductClaimMismatchValidationError(
                f"claim {claim.claim_id!r} has expected_verdict="
                f"{claim.expected_verdict.value!r}, but every product/claim "
                "mismatch claim must be 'incompatible' by construction"
            )


def main() -> None:
    """Validate MISMATCH_CLAIMS_PATH and print a summary."""
    if not MISMATCH_CLAIMS_PATH.exists():
        raise FileNotFoundError(
            f"{MISMATCH_CLAIMS_PATH} does not exist. Run "
            "`make finalize-product-claim-mismatch` first."
        )

    valid_clause_ids = load_valid_clause_ids()
    document_product_lines = load_document_product_lines(MANIFEST_PATH)

    claims = load_claims(MISMATCH_CLAIMS_PATH)
    check_document_ids_exist(claims, set(document_product_lines))
    check_reference_clause_ids_exist(claims, valid_clause_ids)
    check_claim_ids_unique(claims)
    check_targets_non_casco_documents(claims, document_product_lines)
    check_verdict_is_always_incompatible(claims)

    by_product_line: dict[str, int] = {}
    for claim in claims:
        line = document_product_lines.get(claim.document_id, "?")
        by_product_line[line] = by_product_line.get(line, 0) + 1

    print(f"{MISMATCH_CLAIMS_PATH}: {len(claims)} claim(s)")
    print("By product_line:")
    for line, count in sorted(by_product_line.items()):
        print(f"  {line}: {count}")

    print(f"\nTotal: {len(claims)} claim(s). OK.")


if __name__ == "__main__":
    main()
