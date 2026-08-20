#!/usr/bin/env python3
"""Draw the [M1-08] stratified 50-clause sample for parsing-quality review.

This script draws a stratified, reproducible 50-clause sample from the
built corpus (``build/parsed_clauses.jsonl``) and writes it to
``eval/parsing_quality_sample.csv`` with the judgment columns left empty.
Run ``scripts/validate_parsing_quality_sample.py``
(``make validate-parsing-quality-sample``) next to fill those columns via
automated LLM judgment -- using
``document_id``/``filename``/``page_start``/``page_end`` to locate each
clause's source PDF -- then ``scripts/score_parsing_quality.py``
(``make score-parsing-quality``) to score the result; running the scorer
against a file whose judgment columns are still empty is a user error this
script cannot detect for you.

Sampling design: rule-assigned clauses (``type_source=rule``) are taken as
a full census rather than sampled -- every rule-assigned clause is
included, which measures that statistic at full population coverage
instead of an arbitrary subset. The remaining slots are drawn from the
LLM-assigned population via a stratified quota over (product_line, source)
-- see ``QUOTAS`` below for the exact numbers and rationale. Each
multi-era cell's quota is split across the four filing-year "heading era"
buckets (2004-2009, 2010-2016, 2017-2021, 2022-2025) by the
largest-remainder method, then drawn with a fixed seed for reproducibility.

**[M1-08b] (2026-08-19): QUOTAS retuned, 19 -> 14 rule census.** The
[M1-04c]/[M1-08b] boundary and heading-detection fixes (clause
segmentation ``v4`` -> ``v6``) changed which clauses exist and which
titles the deterministic rule pass matches -- the corpus's rule-assigned
population shrank from 19 to 14. ``SAMPLE_SIZE``/``SEED`` are unchanged
per the [M1-08b] DoD ("same seed and stratification design where
possible"), so the Stage-2 top-up grew from 31 to 36 slots to keep the
total at 50: +1 to each of the five non-CASCO/non-CARTA-VERDE cells
(``RCF-A``/text, ``RCF-A``/ocr, ``ASSIST``/text, ``ASSIST``/ocr,
``GAR.EST``/text), leaving CASCO's already-largest quota and CARTA
VERDE's already-thin quota unchanged and preserving both OCR cells'
deliberate oversampling. Because clause boundaries shifted with the
segmentation-version bump, ``clause_id``s in this sample are NOT expected
to match [M1-08]'s original 50-clause sample row-for-row -- this is an
independently-drawn sample under the same design, not a matched-pair
rerun (see ``docs/PARSING.md``'s second-measurement section for the full
discussion).

Known corpus gap, not a sampling artifact: no rule-assigned clause is
OCR-derived (``source=ocr``) anywhere in the corpus, so an "OCR intersected
with rule-assigned" accuracy split cannot be measured from this sample.

**[M1-08c]: `boundary_source` column added, SAMPLE_SIZE/SEED/QUOTAS
unchanged.** The [M1-04d] vision-LLM boundary-escalation pass corrects some
clauses' page ranges in place -- it does not add, remove or reclassify any
clause, so this script's sampling design carries over from [M1-08b]
unmodified. The new ``boundary_source`` system column (``deterministic`` or
``vision_escalated``, from [domain.clause_tree.BoundarySource]) lets
[M1-08c] split accuracy by whether a clause's boundary was touched by that
pass, isolating its effect from ordinary sampling noise. Because escalation
only ever reassigns lines between existing sibling clauses -- never splits
or restructures the tree -- ``clause_id``s are expected to be far more
stable against [M1-08b]'s sample than [M1-08b]'s were against [M1-08]'s;
see ``docs/PARSING.md``'s third-measurement section for the confirmed
overlap count.
"""

from __future__ import annotations

import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

from domain.clause_classification import TypeSource
from infrastructure.parsing.clause_schema import ParsedClauseRecord
from infrastructure.parsing.corpus_artifact import JSONL_PATH, read_parsed_clauses_jsonl
from infrastructure.parsing.manifest import read_manifest

MANIFEST_PATH = Path("data/policies/manifest.csv")
OUTPUT_PATH = Path("eval/parsing_quality_sample.csv")

SAMPLE_SIZE = 50
SEED = 42
TEXT_EXCERPT_CHARS = 1500

ERA_BUCKETS: tuple[tuple[str, int, int], ...] = (
    ("2004-2009", 2004, 2009),
    ("2010-2016", 2010, 2016),
    ("2017-2021", 2017, 2021),
    ("2022-2025", 2022, 2025),
)

# (product_line, source) -> quota for the Stage-2 stratified top-up. Stage 1
# (all rule-assigned clauses -- 14 as of [M1-08b], was 19 at [M1-08]) is a
# census, not drawn from a quota. See the module docstring's [M1-08b] note
# for why these numbers grew from the original 31-total quota.
QUOTAS: dict[tuple[str, str], int] = {
    ("CASCO", "text"): 9,
    ("RCF-A", "text"): 4,
    ("RCF-A", "ocr"): 6,
    ("ASSIST", "text"): 4,
    ("ASSIST", "ocr"): 6,
    ("GAR.EST", "text"): 4,
    ("CARTA VERDE", "text"): 3,
}

JUDGMENT_COLUMNS = (
    "boundary_correct",
    "boundary_notes",
    "reference_clause_type",
    "provenance_correct",
    "provenance_notes",
    "failure_mode_tag",
    "reviewer_notes",
)

SYSTEM_COLUMNS = (
    "sample_id",
    "clause_id",
    "document_id",
    "filename",
    "product_line",
    "insurer",
    "cnpj",
    "indemnity_regime",
    "susep_process",
    "filing_year",
    "era_bucket",
    "page_start",
    "page_end",
    "source",
    "boundary_source",
    "type_source",
    "predicted_clause_type",
    "confidence",
    "path",
    "bundle_section",
    "title",
    "text_excerpt",
    "text_char_count",
)


def era_bucket(filing_year: str) -> str:
    """Map a filing year to one of the four heading-era vintage buckets."""
    year = int(filing_year)
    for label, start, end in ERA_BUCKETS:
        if start <= year <= end:
            return label
    raise ValueError(f"Filing year {filing_year} falls outside all era buckets")


def load_corpus() -> list[ParsedClauseRecord]:
    """Load the built corpus, failing loudly if `make parse` hasn't run yet."""
    if not JSONL_PATH.exists():
        raise FileNotFoundError(
            f"{JSONL_PATH} does not exist. Run `make parse` first to build the corpus."
        )
    return read_parsed_clauses_jsonl(JSONL_PATH)


def load_filename_lookup(manifest_path: Path) -> dict[str, str]:
    """Return a document_id -> filename lookup from manifest.csv."""
    return {row["id"]: row["filename"] for row in read_manifest(manifest_path)}


def allocate_era_quota(era_counts: dict[str, int], quota: int) -> dict[str, int]:
    """Split `quota` across eras proportionally, largest remainder first.

    Never allocates more to an era than that era actually has available.
    """
    total = sum(era_counts.values())
    if total == 0 or quota == 0:
        return dict.fromkeys(era_counts, 0)

    exact = {era: quota * count / total for era, count in era_counts.items()}
    floors = {era: min(int(value), era_counts[era]) for era, value in exact.items()}
    remaining = quota - sum(floors.values())

    by_largest_remainder = sorted(
        (era for era in era_counts if floors[era] < era_counts[era]),
        key=lambda era: exact[era] - floors[era],
        reverse=True,
    )
    allocation = dict(floors)
    for era in by_largest_remainder:
        if remaining <= 0:
            break
        allocation[era] += 1
        remaining -= 1

    return allocation


def draw_stratified_sample(
    records: list[ParsedClauseRecord],
    quotas: dict[tuple[str, str], int],
    seed: int,
) -> list[ParsedClauseRecord]:
    """Draw the Stage-2 stratified top-up from the (LLM-stub) non-rule population."""
    rng = random.Random(seed)
    sampled: list[ParsedClauseRecord] = []

    by_cell: dict[tuple[str, str], list[ParsedClauseRecord]] = defaultdict(list)
    for record in records:
        by_cell[(record.product_line, record.source)].append(record)

    for cell, quota in quotas.items():
        pool = by_cell.get(cell, [])
        if len(pool) < quota:
            raise ValueError(f"Cell {cell} has only {len(pool)} clauses, needs {quota}")

        by_era: dict[str, list[ParsedClauseRecord]] = defaultdict(list)
        for record in pool:
            by_era[era_bucket(record.filing_year)].append(record)

        if len(by_era) == 1:
            sampled.extend(rng.sample(pool, quota))
            continue

        era_counts = {era: len(items) for era, items in by_era.items()}
        era_quota = allocate_era_quota(era_counts, quota)
        for era, count in era_quota.items():
            sampled.extend(rng.sample(by_era[era], count))

    return sampled


def build_annotation_rows(
    records: list[ParsedClauseRecord], filenames: dict[str, str]
) -> list[dict[str, object]]:
    """Sort the sampled records and build one annotation row per clause."""
    ordered = sorted(records, key=lambda r: (int(r.document_id), r.page_start, r.path))

    rows: list[dict[str, object]] = []
    for sample_id, record in enumerate(ordered, start=1):
        text = record.text
        excerpt = text[:TEXT_EXCERPT_CHARS]
        if len(text) > TEXT_EXCERPT_CHARS:
            excerpt += f"... [truncated, {len(text)} chars total]"

        row: dict[str, object] = {
            "sample_id": sample_id,
            "clause_id": record.clause_id,
            "document_id": record.document_id,
            "filename": filenames[record.document_id],
            "product_line": record.product_line,
            "insurer": record.insurer,
            "cnpj": record.cnpj,
            "indemnity_regime": record.indemnity_regime,
            "susep_process": record.susep_process,
            "filing_year": record.filing_year,
            "era_bucket": era_bucket(record.filing_year),
            "page_start": record.page_start,
            "page_end": record.page_end,
            "source": record.source,
            "boundary_source": record.boundary_source or "",
            "type_source": record.type_source.value,
            "predicted_clause_type": record.clause_type.value,
            "confidence": record.confidence,
            "path": record.path,
            "bundle_section": record.bundle_section or "",
            "title": record.title,
            "text_excerpt": excerpt,
            "text_char_count": len(text),
        }
        for column in JUDGMENT_COLUMNS:
            row[column] = ""
        rows.append(row)
    return rows


def write_annotation_csv(rows: list[dict[str, object]], path: Path) -> None:
    """Write the annotation rows to CSV, system columns first, judgments last."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(SYSTEM_COLUMNS) + list(JUDGMENT_COLUMNS)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, object]]) -> None:
    """Print a stratification summary of the drawn sample to stdout."""
    print(f"Sample size: {len(rows)}")
    for label in ("product_line", "source", "era_bucket", "type_source"):
        counts = Counter(row[label] for row in rows)
        print(f"  by {label}: {dict(sorted(counts.items()))}")


def main() -> None:
    """Draw the M1-08 stratified sample and write it to OUTPUT_PATH."""
    records = load_corpus()
    filenames = load_filename_lookup(MANIFEST_PATH)

    rule_records = [r for r in records if r.type_source == TypeSource.RULE]
    llm_records = [r for r in records if r.type_source == TypeSource.LLM]

    top_up = draw_stratified_sample(llm_records, QUOTAS, SEED)
    sample = rule_records + top_up

    if len(sample) != SAMPLE_SIZE:
        raise ValueError(
            f"Expected {SAMPLE_SIZE} sampled clauses, got {len(sample)} "
            f"({len(rule_records)} rule census + {len(top_up)} stratified top-up). "
            "The corpus has likely changed since QUOTAS was tuned -- re-derive it."
        )

    product_lines = {r.product_line for r in sample}
    if len(product_lines) != 5:
        raise ValueError(
            f"Sample covers only {len(product_lines)} product lines: "
            f"{sorted(product_lines)}"
        )

    sources = {r.source for r in sample}
    if sources != {"text", "ocr"}:
        raise ValueError(
            f"Sample does not cover both extraction modes: {sorted(sources)}"
        )

    eras = {era_bucket(r.filing_year) for r in sample}
    if len(eras) != 4:
        raise ValueError(f"Sample covers only {len(eras)} heading eras: {sorted(eras)}")

    rows = build_annotation_rows(sample, filenames)
    write_annotation_csv(rows, OUTPUT_PATH)
    summarize(rows)
    print(f"Wrote {len(rows)} clauses to {OUTPUT_PATH}.")


if __name__ == "__main__":
    main()
