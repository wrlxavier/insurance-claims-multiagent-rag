#!/usr/bin/env python3
"""Score the validated [M1-08] sample and publish the results.

See ``scripts/sample_parsing_quality.py`` for how the sample is drawn and
its module docstring for the sampling design, and
``scripts/validate_parsing_quality_sample.py`` for how the judgment columns
are filled via automated LLM validation. This script reads the now-
validated ``eval/parsing_quality_sample.csv``, computes boundary accuracy,
type accuracy, a confusion matrix, accuracy split by ``type_source`` (rule
vs LLM-stub), by ``source`` (text vs OCR) and by ``boundary_source``
(deterministic vs [M1-04d]'s vision-escalated pass, added for [M1-08c]),
provenance accuracy, and a failure-mode frequency table, then writes
``eval/parsing_quality_results.md``. That file -- together with the
validated CSV -- is the evaluation data the [M1-08] DoD asks to be
committed; ``docs/PARSING.md`` copies its tables rather than recomputing
them by hand.

``type_correct`` is not a column in the CSV -- it is derived here as
``predicted_clause_type == reference_clause_type``, since the reviewer only
records the reference type, not a redundant correct/incorrect flag.
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from domain.clause_classification import ClauseType

INPUT_PATH = Path("eval/parsing_quality_sample.csv")
OUTPUT_PATH = Path("eval/parsing_quality_results.md")

CLAUSE_TYPES = [clause_type.value for clause_type in ClauseType]

_TRUE_VALUES = {"true", "y", "yes", "1"}
_FALSE_VALUES = {"false", "n", "no", "0"}


def read_annotations(path: Path) -> list[dict[str, str]]:
    """Read the annotated sample, failing loudly if it hasn't been drawn yet."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Run `make sample-parsing-quality` first."
        )
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_bool(value: str) -> bool:
    """Normalize a hand-entered boolean cell (TRUE/FALSE/Y/N/yes/no/1/0)."""
    normalized = value.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise ValueError(f"Cannot parse {value!r} as a boolean")


def validate_annotations(rows: list[dict[str, str]]) -> None:
    """Fail loudly, listing every row with a blank or invalid judgment column.

    A partially annotated file is a user error this script must catch
    rather than silently score against blanks.
    """
    problems: list[str] = []
    for row in rows:
        sample_id = row["sample_id"]

        for bool_column in ("boundary_correct", "provenance_correct"):
            try:
                parse_bool(row[bool_column])
            except ValueError:
                problems.append(
                    f"sample_id={sample_id}: invalid {bool_column}={row[bool_column]!r}"
                )

        reference_type = row["reference_clause_type"].strip()
        if reference_type not in CLAUSE_TYPES:
            problems.append(
                f"sample_id={sample_id}: invalid reference_clause_type="
                f"{row['reference_clause_type']!r} (expected one of {CLAUSE_TYPES})"
            )

    if problems:
        raise ValueError(
            "eval/parsing_quality_sample.csv is not fully/correctly annotated:\n"
            + "\n".join(problems)
        )


def is_type_correct(row: dict[str, str]) -> bool:
    """Return whether the predicted type matches the reviewer's reference type."""
    return row["predicted_clause_type"] == row["reference_clause_type"]


def _accuracy(flags: list[bool]) -> tuple[float, int]:
    """Return (accuracy, n) for a list of correctness flags."""
    if not flags:
        return 0.0, 0
    return sum(flags) / len(flags), len(flags)


def compute_boundary_accuracy(rows: list[dict[str, str]]) -> tuple[float, int]:
    """Overall boundary accuracy across the whole sample."""
    return _accuracy([parse_bool(row["boundary_correct"]) for row in rows])


def compute_type_accuracy(rows: list[dict[str, str]]) -> tuple[float, int]:
    """Overall type accuracy across the whole sample."""
    return _accuracy([is_type_correct(row) for row in rows])


def compute_provenance_accuracy(rows: list[dict[str, str]]) -> tuple[float, int]:
    """Overall provenance accuracy across the whole sample."""
    return _accuracy([parse_bool(row["provenance_correct"]) for row in rows])


def compute_confusion_matrix(rows: list[dict[str, str]]) -> dict[str, dict[str, int]]:
    """Reference type (rows) x predicted type (columns), all 6 types on both axes."""
    matrix = {ref: dict.fromkeys(CLAUSE_TYPES, 0) for ref in CLAUSE_TYPES}
    for row in rows:
        matrix[row["reference_clause_type"]][row["predicted_clause_type"]] += 1
    return matrix


def compute_accuracy_by_type_source(
    rows: list[dict[str, str]],
) -> dict[str, dict[str, tuple[float, int]]]:
    """Boundary and type accuracy split by type_source (rule vs llm)."""
    result: dict[str, dict[str, tuple[float, int]]] = {}
    for type_source in ("rule", "llm"):
        subset = [row for row in rows if row["type_source"] == type_source]
        result[type_source] = {
            "boundary": compute_boundary_accuracy(subset),
            "type": compute_type_accuracy(subset),
        }
    return result


def compute_accuracy_by_boundary_source(
    rows: list[dict[str, str]],
) -> dict[str, dict[str, tuple[float, int]]]:
    """Boundary/type accuracy split by boundary_source (deterministic vs escalated)."""
    result: dict[str, dict[str, tuple[float, int]]] = {}
    for boundary_source in ("deterministic", "vision_escalated"):
        subset = [row for row in rows if row["boundary_source"] == boundary_source]
        result[boundary_source] = {
            "boundary": compute_boundary_accuracy(subset),
            "type": compute_type_accuracy(subset),
        }
    return result


def compute_accuracy_by_extraction_mode(
    rows: list[dict[str, str]],
) -> dict[str, dict[str, tuple[float, int]]]:
    """Boundary and type accuracy split by source (text vs ocr)."""
    result: dict[str, dict[str, tuple[float, int]]] = {}
    for source in ("text", "ocr"):
        subset = [row for row in rows if row["source"] == source]
        result[source] = {
            "boundary": compute_boundary_accuracy(subset),
            "type": compute_type_accuracy(subset),
        }
    return result


def collect_failure_examples(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Every row with at least one incorrect judgment, in sample_id order."""
    failures = []
    for row in rows:
        if (
            not parse_bool(row["boundary_correct"])
            or not is_type_correct(row)
            or not parse_bool(row["provenance_correct"])
        ):
            failures.append(row)
    return failures


def failure_mode_frequency(failures: list[dict[str, str]]) -> Counter[str]:
    """Count failure_mode_tag across failing rows only, most common first."""
    return Counter(
        row["failure_mode_tag"] for row in failures if row["failure_mode_tag"]
    )


def _fmt_pct(accuracy: float, n: int) -> str:
    """Render an (accuracy, n) pair as e.g. '86.0% (43/50)'."""
    correct = round(accuracy * n)
    return f"{accuracy:.1%} ({correct}/{n})"


def render_markdown_report(
    rows: list[dict[str, str]],
    boundary: tuple[float, int],
    type_acc: tuple[float, int],
    provenance: tuple[float, int],
    confusion: dict[str, dict[str, int]],
    by_type_source: dict[str, dict[str, tuple[float, int]]],
    by_boundary_source: dict[str, dict[str, tuple[float, int]]],
    by_extraction_mode: dict[str, dict[str, tuple[float, int]]],
    failures: list[dict[str, str]],
    failure_tags: Counter[str],
) -> str:
    """Render the Markdown results report."""
    lines = [
        "# Parsing quality — scored results",
        "",
        "Generated by `scripts/score_parsing_quality.py` from the validated "
        "50-clause sample in `eval/parsing_quality_sample.csv` (drawn by "
        "`scripts/sample_parsing_quality.py`, stratified across product line, "
        "extraction mode and filing-year era, then validated by "
        "`scripts/validate_parsing_quality_sample.py`'s automated LLM judgment). "
        "This is the evaluation data behind `docs/PARSING.md`, not just a table "
        "in that document -- every number here is reproducible by re-running "
        "`make score-parsing-quality` against the same validated CSV.",
        "",
        "## Overall accuracy",
        "",
        f"- Boundary accuracy: {_fmt_pct(*boundary)}",
        f"- Type accuracy: {_fmt_pct(*type_acc)}",
        f"- Provenance accuracy: {_fmt_pct(*provenance)}",
        "",
        "## Confusion matrix (reference type x predicted type)",
        "",
        "| reference \\ predicted | " + " | ".join(CLAUSE_TYPES) + " |",
        "| --- | " + " | ".join(["---:"] * len(CLAUSE_TYPES)) + " |",
    ]
    for reference_type in CLAUSE_TYPES:
        row_counts = confusion[reference_type]
        lines.append(
            f"| {reference_type} | "
            + " | ".join(str(row_counts[predicted]) for predicted in CLAUSE_TYPES)
            + " |"
        )

    lines += [
        "",
        "## Accuracy by type_source",
        "",
        "| type_source | n | boundary accuracy | type accuracy |",
        "| --- | ---: | ---: | ---: |",
    ]
    for type_source, metrics in by_type_source.items():
        n = metrics["boundary"][1]
        lines.append(
            f"| {type_source} | {n} | {_fmt_pct(*metrics['boundary'])} | "
            f"{_fmt_pct(*metrics['type'])} |"
        )

    lines += [
        "",
        "## Accuracy by boundary_source",
        "",
        "| boundary_source | n | boundary accuracy | type accuracy |",
        "| --- | ---: | ---: | ---: |",
    ]
    for boundary_source, metrics in by_boundary_source.items():
        n = metrics["boundary"][1]
        lines.append(
            f"| {boundary_source} | {n} | {_fmt_pct(*metrics['boundary'])} | "
            f"{_fmt_pct(*metrics['type'])} |"
        )

    lines += [
        "",
        "## Accuracy by extraction mode",
        "",
        "| source | n | boundary accuracy | type accuracy |",
        "| --- | ---: | ---: | ---: |",
    ]
    for source, metrics in by_extraction_mode.items():
        n = metrics["boundary"][1]
        lines.append(
            f"| {source} | {n} | {_fmt_pct(*metrics['boundary'])} | "
            f"{_fmt_pct(*metrics['type'])} |"
        )

    lines += [
        "",
        "## Failure mode frequency",
        "",
        f"Across {len(failures)} of {len(rows)} sampled clauses with at least one "
        "incorrect judgment (boundary, type or provenance):",
        "",
        "| failure_mode_tag | count |",
        "| --- | ---: |",
    ]
    for tag, count in failure_tags.most_common():
        lines.append(f"| {tag} | {count} |")

    lines += [
        "",
        "## Failure examples",
        "",
        "| sample_id | clause_id | document | pages | predicted type | "
        "reference type | boundary ok | provenance ok | tag | notes |",
        "| --- | --- | --- | ---: | --- | --- | --- | --- | --- | --- |",
    ]
    for row in failures:
        notes = " ".join(
            note
            for note in (
                row["boundary_notes"],
                row["provenance_notes"],
                row["reviewer_notes"],
            )
            if note
        )
        lines.append(
            f"| {row['sample_id']} | `{row['clause_id']}` | {row['document_id']} "
            f"({row['filename']}) | {row['page_start']}-{row['page_end']} | "
            f"{row['predicted_clause_type']} | {row['reference_clause_type']} | "
            f"{row['boundary_correct']} | {row['provenance_correct']} | "
            f"{row['failure_mode_tag']} | {notes} |"
        )

    lines += [
        "",
        "## Summary",
        "",
        f"- Sample size: {len(rows)}",
        f"- Boundary accuracy: {_fmt_pct(*boundary)}",
        f"- Type accuracy: {_fmt_pct(*type_acc)}",
        f"- Provenance accuracy: {_fmt_pct(*provenance)}",
        f"- Rows with at least one failure: {len(failures)}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    """Score the annotated sample and write the results report."""
    rows = read_annotations(INPUT_PATH)
    validate_annotations(rows)

    boundary = compute_boundary_accuracy(rows)
    type_acc = compute_type_accuracy(rows)
    provenance = compute_provenance_accuracy(rows)
    confusion = compute_confusion_matrix(rows)
    by_type_source = compute_accuracy_by_type_source(rows)
    by_boundary_source = compute_accuracy_by_boundary_source(rows)
    by_extraction_mode = compute_accuracy_by_extraction_mode(rows)
    failures = collect_failure_examples(rows)
    failure_tags = failure_mode_frequency(failures)

    report = render_markdown_report(
        rows,
        boundary,
        type_acc,
        provenance,
        confusion,
        by_type_source,
        by_boundary_source,
        by_extraction_mode,
        failures,
        failure_tags,
    )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(report, encoding="utf-8")

    print(f"Boundary accuracy: {_fmt_pct(*boundary)}")
    print(f"Type accuracy: {_fmt_pct(*type_acc)}")
    print(f"Provenance accuracy: {_fmt_pct(*provenance)}")
    print(f"Failing rows: {len(failures)}/{len(rows)}")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
