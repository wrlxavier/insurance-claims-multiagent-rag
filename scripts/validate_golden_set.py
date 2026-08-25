#!/usr/bin/env python3
"""Validate the golden set against its schema and the parsed corpus [M2-01].

Loads every ``data/golden_set/*.jsonl`` file, validates each row against
[infrastructure.evaluation.golden_set_schema.GoldenQuestion] (schema,
enum vocabulary, and the unanswerable-consistency rule all fail loudly via
Pydantic), and checks three structural properties a schema alone can't
express:

- **one file per question type**: every row in ``<type>.jsonl`` must have
  ``question_type == <type>``.
- **document_id exists**: every row's ``document_id`` must be a real row in
  ``data/policies/manifest.csv``.
- **reference_clause_ids exist**: every id must be a real ``clause_id`` in
  the built corpus (``build/parsed_clauses.jsonl``) -- this is the DoD's
  core check, and the one CI runs so a re-parse that breaks a clause id
  fails visibly. Run ``make fetch-corpus-artifacts`` (or ``make parse``) to
  produce that file if it's missing.

Also asserts ``question_id`` is unique across the whole golden set.

Run via ``make validate-golden-set``.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from infrastructure.evaluation.golden_set_schema import GoldenQuestion, QuestionType
from infrastructure.parsing.corpus_artifact import JSONL_PATH, read_parsed_clauses_jsonl
from infrastructure.parsing.manifest import read_manifest

GOLDEN_SET_DIR = Path("data/golden_set")
MANIFEST_PATH = Path("data/policies/manifest.csv")


class GoldenSetValidationError(Exception):
    """Raised when the golden set fails a structural or schema check."""


def load_valid_clause_ids() -> set[str]:
    """Return the set of every clause_id in the built corpus."""
    if not JSONL_PATH.exists():
        raise FileNotFoundError(
            f"{JSONL_PATH} does not exist. Run `make fetch-corpus-artifacts` "
            "(pre-built corpus) or `make parse` (full rebuild) first."
        )
    return {record.clause_id for record in read_parsed_clauses_jsonl(JSONL_PATH)}


def load_valid_document_ids(manifest_path: Path) -> set[str]:
    """Return the set of every document_id in manifest.csv."""
    return {row["id"] for row in read_manifest(manifest_path)}


def load_questions(path: Path) -> list[GoldenQuestion]:
    """Parse every line of one golden-set JSONL file as a GoldenQuestion."""
    questions = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                questions.append(GoldenQuestion.model_validate(json.loads(line)))
            except (json.JSONDecodeError, ValidationError) as error:
                raise GoldenSetValidationError(
                    f"{path}:{line_number}: invalid golden question -- {error}"
                ) from error
    return questions


def check_question_type_matches_filename(
    path: Path, questions: list[GoldenQuestion]
) -> None:
    """Assert every row's question_type matches the file it lives in."""
    try:
        expected_type = QuestionType(path.stem)
    except ValueError as error:
        raise GoldenSetValidationError(
            f"{path}: filename {path.stem!r} is not a valid question_type"
        ) from error
    for question in questions:
        if question.question_type != expected_type:
            raise GoldenSetValidationError(
                f"{path}: question {question.question_id!r} has "
                f"question_type={question.question_type.value!r}, expected "
                f"{expected_type.value!r} to match the filename"
            )


def check_document_ids_exist(
    path: Path, questions: list[GoldenQuestion], valid_document_ids: set[str]
) -> None:
    """Assert every row's document_id exists in manifest.csv."""
    for question in questions:
        if question.document_id not in valid_document_ids:
            raise GoldenSetValidationError(
                f"{path}: question {question.question_id!r} references unknown "
                f"document_id {question.document_id!r}"
            )


def check_reference_clause_ids_exist(
    path: Path, questions: list[GoldenQuestion], valid_clause_ids: set[str]
) -> None:
    """Assert every row's reference_clause_ids all exist in the corpus."""
    for question in questions:
        unknown = [
            clause_id
            for clause_id in question.reference_clause_ids
            if clause_id not in valid_clause_ids
        ]
        if unknown:
            raise GoldenSetValidationError(
                f"{path}: question {question.question_id!r} references "
                f"unknown clause_id(s) {unknown!r}"
            )


def check_question_ids_unique(all_questions: list[tuple[Path, GoldenQuestion]]) -> None:
    """Assert question_id is unique across every golden-set file."""
    seen: dict[str, Path] = {}
    for path, question in all_questions:
        if question.question_id in seen:
            raise GoldenSetValidationError(
                f"{path}: duplicate question_id {question.question_id!r} "
                f"(first seen in {seen[question.question_id]})"
            )
        seen[question.question_id] = path


def main() -> None:
    """Validate every file under data/golden_set/ and print a summary."""
    valid_clause_ids = load_valid_clause_ids()
    valid_document_ids = load_valid_document_ids(MANIFEST_PATH)

    paths = sorted(GOLDEN_SET_DIR.glob("*.jsonl"))
    if not paths:
        raise GoldenSetValidationError(f"No JSONL files found under {GOLDEN_SET_DIR}")

    all_questions: list[tuple[Path, GoldenQuestion]] = []
    for path in paths:
        questions = load_questions(path)
        check_question_type_matches_filename(path, questions)
        check_document_ids_exist(path, questions, valid_document_ids)
        check_reference_clause_ids_exist(path, questions, valid_clause_ids)
        all_questions.extend((path, question) for question in questions)
        print(f"{path}: {len(questions)} question(s)")

    check_question_ids_unique(all_questions)

    print(f"Total: {len(all_questions)} question(s) across {len(paths)} file(s). OK.")


if __name__ == "__main__":
    main()
