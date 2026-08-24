#!/usr/bin/env python3
"""Promote human-approved draft rows into the synthetic claim set [M2-04].

Mirrors ``scripts/finalize_golden_set_from_review.py``'s flow exactly: the
author's review happens on ``eval/synthetic_claims_draft.csv`` (edit any
column, set ``approved=Y`` per row accepted) before this script ever runs.
It only reads that decision -- it does not judge narrative quality itself.

For every row with ``approved`` truthy and no ``finalized_claim_id`` yet, the
PII safety net ([pii_safety_net.scan_narrative_for_pii]) runs a second time
as a gate: a hit here blocks that row's promotion rather than silently
carrying a flagged narrative into the permanent record. Surviving rows get
the next sequential ``claim_id`` for their ``expected_verdict`` (continuing
from whatever is already in ``data/synthetic_claims/claims.jsonl``, never
assuming a fresh start), are validated through
[infrastructure.evaluation.synthetic_claims_schema.SyntheticClaim], appended,
and the assigned id is written back into the CSV's ``finalized_claim_id``
column. Finally runs ``scripts/validate_synthetic_claims.py``.

Run via ``make finalize-synthetic-claims``.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections.abc import Sequence
from datetime import UTC, date, datetime
from pathlib import Path

from pydantic import ValidationError

from infrastructure.evaluation.synthetic_claims_schema import (
    SCHEMA_VERSION,
    SyntheticClaim,
)

try:
    # Direct execution: the script's own directory is sys.path[0].
    from pii_safety_net import scan_narrative_for_pii
    from validate_synthetic_claims import main as run_validate_synthetic_claims
except ModuleNotFoundError:
    # Imported as a package (pytest, repo root on sys.path).
    from scripts.pii_safety_net import scan_narrative_for_pii
    from scripts.validate_synthetic_claims import main as run_validate_synthetic_claims

DEFAULT_CSV_PATH = Path("eval/synthetic_claims_draft.csv")
CLAIMS_PATH = Path("data/synthetic_claims/claims.jsonl")

TRUTHY_APPROVED = {"y", "yes", "true", "1"}

_CLAIM_ID_PATTERN = re.compile(r"^(?P<verdict>[a-z_]+)-(?P<seq>\d{3})$")


class FinalizeError(Exception):
    """Raised when a row can't be turned into a valid SyntheticClaim."""


def is_approved(row: dict[str, str]) -> bool:
    """Whether a CSV row's `approved` column is a truthy value."""
    return row.get("approved", "").strip().lower() in TRUTHY_APPROVED


def next_sequence_numbers(claims_path: Path) -> dict[str, int]:
    """Return, per expected_verdict, the next unused 3-digit sequence number."""
    max_seq: dict[str, int] = {}
    if not claims_path.exists():
        return max_seq
    with claims_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            claim_id = json.loads(line)["claim_id"]
            match = _CLAIM_ID_PATTERN.match(claim_id)
            if match:
                verdict = match.group("verdict")
                seq = int(match.group("seq"))
                max_seq[verdict] = max(max_seq.get(verdict, 0), seq)
    return {verdict: seq + 1 for verdict, seq in max_seq.items()}


def row_to_synthetic_claim(
    row: dict[str, str], *, claim_id: str, authored_at: date
) -> SyntheticClaim:
    """Validate one approved CSV row into a SyntheticClaim, assigning its identity.

    ``notes`` prefers the reviewer's own ``review_correction`` over the LLM's
    ``draft_notes`` -- same reasoning as
    ``finalize_golden_set_from_review.row_to_golden_question``: defaulting to
    the draft would make the finalized rationale indistinguishable from the
    model's, when the whole point is to record the author's own call.
    """
    reference_clause_ids = [
        cid for cid in row["reference_clause_ids"].split(";") if cid
    ]
    row_authored_at = row.get("authored_at", "").strip()
    missing_fact_type = row.get("missing_fact_type", "").strip() or None
    try:
        return SyntheticClaim.model_validate(
            {
                "schema_version": SCHEMA_VERSION,
                "claim_id": claim_id,
                "document_id": row["document_id"],
                "narrative": row["narrative"],
                "reference_clause_ids": reference_clause_ids,
                "expected_verdict": row["expected_verdict"],
                "missing_fact_type": missing_fact_type,
                "notes": row.get("review_correction", "").strip(),
                "authored_at": row_authored_at or authored_at.isoformat(),
            }
        )
    except ValidationError as error:
        raise FinalizeError(f"row_id={row['row_id']!r}: {error}") from error


def finalize_rows(
    rows: list[dict[str, str]], *, authored_at: date
) -> tuple[list[dict[str, str]], list[SyntheticClaim], list[str]]:
    """Assign claim_ids and build a SyntheticClaim per unfinalized approved row.

    Returns the (possibly updated) full row list, the new claims to append,
    and the row_ids skipped due to a PII safety-net hit.
    """
    next_seq = next_sequence_numbers(CLAIMS_PATH)
    new_claims: list[SyntheticClaim] = []
    pii_blocked: list[str] = []

    for row in rows:
        if not is_approved(row) or row.get("finalized_claim_id", "").strip():
            continue
        if scan_narrative_for_pii(row["narrative"]):
            pii_blocked.append(row["row_id"])
            continue
        verdict = row["expected_verdict"].strip()
        seq = next_seq.get(verdict, 1)
        claim_id = f"{verdict}-{seq:03d}"
        claim = row_to_synthetic_claim(row, claim_id=claim_id, authored_at=authored_at)
        next_seq[verdict] = seq + 1
        new_claims.append(claim)
        row["finalized_claim_id"] = claim_id

    return rows, new_claims, pii_blocked


def append_claims(claims: list[SyntheticClaim]) -> None:
    """Append newly finalized claims to data/synthetic_claims/claims.jsonl."""
    CLAIMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CLAIMS_PATH.open("a", encoding="utf-8") as handle:
        for claim in claims:
            handle.write(claim.model_dump_json())
            handle.write("\n")


def write_csv(
    path: Path, fieldnames: Sequence[str], rows: list[dict[str, str]]
) -> None:
    """Rewrite the draft CSV with finalized_claim_id filled in for new rows."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(csv_path: Path = DEFAULT_CSV_PATH) -> None:
    """Finalize every approved, not-yet-finalized row in `csv_path`."""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} does not exist. Run `make draft-synthetic-claims` first."
        )

    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if fieldnames is None:
        raise ValueError(f"{csv_path} has no header row.")

    authored_at = datetime.now(UTC).date()
    rows, new_claims, pii_blocked = finalize_rows(rows, authored_at=authored_at)

    if pii_blocked:
        print(
            f"WARNING: {len(pii_blocked)} approved row(s) blocked by the PII "
            f"safety net -- not finalized: {', '.join(pii_blocked)}",
            file=sys.stderr,
        )

    if not new_claims:
        print(
            f"No newly-approved, unfinalized rows found in {csv_path}. Nothing to do."
        )
        return

    append_claims(new_claims)
    print(f"Appended {len(new_claims)} claim(s) to {CLAIMS_PATH}")

    write_csv(csv_path, fieldnames, rows)
    print(f"Wrote finalized_claim_id back into {csv_path}")

    print("\nRunning `make validate-synthetic-claims` equivalent...")
    try:
        run_validate_synthetic_claims()
    except Exception as error:
        print(f"VALIDATION FAILED: {error}", file=sys.stderr)
        raise

    by_verdict: dict[str, int] = {}
    for claim in new_claims:
        by_verdict[claim.expected_verdict.value] = (
            by_verdict.get(claim.expected_verdict.value, 0) + 1
        )
    print(f"\nFinalized {len(new_claims)} new claim(s):")
    for verdict, count in sorted(by_verdict.items()):
        print(f"  {verdict}: {count}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help=f"Draft CSV to finalize (default: {DEFAULT_CSV_PATH}).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(csv_path=_parse_args().csv)
