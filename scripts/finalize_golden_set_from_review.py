#!/usr/bin/env python3
"""Promote human-approved draft rows into the real golden set [M2-02].

The author's verification step in [M2-01]'s three-layer authoring flow
happens on ``eval/golden_set_draft_casco.csv`` before this script ever
runs (see ``scripts/draft_golden_questions_casco.py``'s module docstring):
the reviewer edits the LLM-drafted ``question``/``reference_clause_ids``/
etc. columns as needed and sets ``approved=Y`` per row they accept. This
script only reads that decision -- it does not judge correctness or
completeness itself.

For every row with ``approved`` truthy and no ``finalized_question_id`` yet
(so reruns never double-finalize a row), assigns the next sequential
``question_id`` for that row's ``question_type`` -- continuing from
whatever is already in ``data/golden_set/<type>.jsonl``, never assuming a
fresh start, since this script may be rerun later against other draft CSVs
(e.g. M2-03's, via ``--csv``) -- fills ``authored_at`` from the row, falling
back to today's date (see [row_to_golden_question]), validates the row
through [infrastructure.evaluation.golden_set_schema.GoldenQuestion], appends
it to the right JSONL file, and writes the assigned id back into the CSV's
``finalized_question_id`` column. Finally runs
``scripts/validate_golden_set.py`` and reports pass/fail.

Run via ``make finalize-golden-set-casco`` (default CSV) or
``make finalize-golden-set-adversarial`` (``--csv
eval/golden_set_draft_adversarial.csv``).
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

from infrastructure.evaluation.golden_set_schema import SCHEMA_VERSION, GoldenQuestion

try:
    # Direct execution (`python scripts/finalize_golden_set_from_review.py`,
    # as `make finalize-golden-set-casco` does): the script's own directory
    # is sys.path[0], so the sibling module resolves by its bare name.
    from validate_golden_set import main as run_validate_golden_set
except ModuleNotFoundError:
    # Imported as a package (e.g. by pytest, with the repo root on
    # sys.path): only the fully-qualified name resolves.
    from scripts.validate_golden_set import main as run_validate_golden_set

DEFAULT_CSV_PATH = Path("eval/golden_set_draft_casco.csv")
GOLDEN_SET_DIR = Path("data/golden_set")

TRUTHY_APPROVED = {"y", "yes", "true", "1"}

_QUESTION_ID_PATTERN = re.compile(r"^(?P<type>[a-z_]+)-(?P<seq>\d{3})$")


class FinalizeError(Exception):
    """Raised when a row can't be turned into a valid GoldenQuestion."""


def is_approved(row: dict[str, str]) -> bool:
    """Whether a CSV row's `approved` column is a truthy value."""
    return row.get("approved", "").strip().lower() in TRUTHY_APPROVED


def next_sequence_numbers(golden_set_dir: Path) -> dict[str, int]:
    """Return, per question_type, the next unused 3-digit sequence number."""
    next_seq: dict[str, int] = {}
    for path in golden_set_dir.glob("*.jsonl"):
        question_type = path.stem
        max_seq = 0
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                question_id = json.loads(line)["question_id"]
                match = _QUESTION_ID_PATTERN.match(question_id)
                if match and match.group("type") == question_type:
                    max_seq = max(max_seq, int(match.group("seq")))
        next_seq[question_type] = max_seq + 1
    return next_seq


def row_to_golden_question(
    row: dict[str, str], *, question_id: str, authored_at: date
) -> GoldenQuestion:
    """Validate one approved CSV row into a GoldenQuestion, assigning its identity.

    ``authored_at`` prefers the row's own value over the run date: a review
    spread over several days should record when the author actually accepted
    each question, which is what [M2-07]'s ">=14 days later" blind
    re-labelling window is measured from.

    ``notes`` is written exactly as the reviewer left it, with no fallback to
    the LLM's ``draft_notes``: defaulting one to the other would make the
    golden set's stated rationale indistinguishable from the draft's, and the
    whole point of the column is to record the author's reasoning, not the
    model's. An empty ``notes`` is therefore left empty, and warned about.
    """
    expected_verdict = row["expected_verdict"].strip() or None
    reference_clause_ids = [
        cid for cid in row["reference_clause_ids"].split(";") if cid
    ]
    row_authored_at = row.get("authored_at", "").strip()
    try:
        return GoldenQuestion.model_validate(
            {
                "schema_version": SCHEMA_VERSION,
                "question_id": question_id,
                "document_id": row["document_id"],
                "question": row["question"],
                "reference_clause_ids": reference_clause_ids,
                "question_type": row["question_type"],
                "difficulty": row["difficulty"],
                "expected_verdict": expected_verdict,
                "notes": row["notes"],
                "authored_at": row_authored_at or authored_at.isoformat(),
            }
        )
    except ValidationError as error:
        raise FinalizeError(f"row_id={row['row_id']!r}: {error}") from error


def finalize_rows(
    rows: list[dict[str, str]], *, authored_at: date
) -> tuple[list[dict[str, str]], dict[str, list[GoldenQuestion]]]:
    """Assign question_ids and build a GoldenQuestion per unfinalized approved row.

    Returns the (possibly updated) full row list and the new questions to
    append, grouped by question_type.
    """
    next_seq = next_sequence_numbers(GOLDEN_SET_DIR)
    new_questions_by_type: dict[str, list[GoldenQuestion]] = {}

    for row in rows:
        if not is_approved(row) or row.get("finalized_question_id", "").strip():
            continue
        question_type = row["question_type"].strip()
        seq = next_seq.get(question_type, 1)
        question_id = f"{question_type}-{seq:03d}"
        question = row_to_golden_question(
            row, question_id=question_id, authored_at=authored_at
        )
        next_seq[question_type] = seq + 1
        new_questions_by_type.setdefault(question_type, []).append(question)
        row["finalized_question_id"] = question_id

    return rows, new_questions_by_type


def append_questions(question_type: str, questions: list[GoldenQuestion]) -> None:
    """Append newly finalized questions to data/golden_set/<type>.jsonl."""
    path = GOLDEN_SET_DIR / f"{question_type}.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        for question in questions:
            handle.write(question.model_dump_json())
            handle.write("\n")


def write_csv(
    path: Path, fieldnames: Sequence[str], rows: list[dict[str, str]]
) -> None:
    """Rewrite the draft CSV with finalized_question_id filled in for new rows."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(csv_path: Path = DEFAULT_CSV_PATH) -> None:
    """Finalize every approved, not-yet-finalized row in `csv_path`."""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} does not exist. Run `make draft-golden-questions-casco` first."
        )

    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if fieldnames is None:
        raise ValueError(f"{csv_path} has no header row.")

    unexplained = [
        row["row_id"]
        for row in rows
        if is_approved(row)
        and not row.get("finalized_question_id", "").strip()
        and not row.get("notes", "").strip()
    ]
    if unexplained:
        print(
            f"WARNING: {len(unexplained)} approved row(s) have an empty `notes` "
            "-- the golden set will record no author rationale for them: "
            f"{', '.join(unexplained[:10])}"
            f"{' ...' if len(unexplained) > 10 else ''}",
            file=sys.stderr,
        )

    authored_at = datetime.now(UTC).date()
    rows, new_questions_by_type = finalize_rows(rows, authored_at=authored_at)

    total_new = sum(len(questions) for questions in new_questions_by_type.values())
    if total_new == 0:
        print(
            f"No newly-approved, unfinalized rows found in {csv_path}. Nothing to do."
        )
        return

    for question_type, questions in new_questions_by_type.items():
        append_questions(question_type, questions)
        target_path = GOLDEN_SET_DIR / f"{question_type}.jsonl"
        print(f"Appended {len(questions)} question(s) to {target_path}")

    write_csv(csv_path, fieldnames, rows)
    print(f"Wrote finalized_question_id back into {csv_path}")

    print("\nRunning `make validate-golden-set` equivalent...")
    try:
        run_validate_golden_set()
    except Exception as error:
        print(f"VALIDATION FAILED: {error}", file=sys.stderr)
        raise

    print(f"\nFinalized {total_new} new question(s):")
    for question_type, questions in sorted(new_questions_by_type.items()):
        print(f"  {question_type}: {len(questions)}")


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
