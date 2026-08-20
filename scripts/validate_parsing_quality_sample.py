#!/usr/bin/env python3
"""Automated LLM validation of the [M1-08b] parsing-quality sample.

[M1-08]'s DoD wrote this step as "manually validate the new sample against
source PDFs" -- for this project, that manual pass is replaced end to end
by ``google/gemini-3.7-flash`` (pinned to the ``google-vertex`` OpenRouter
route, no fallback) looking at the actual rasterized PDF pages, by explicit
project-owner decision: there is no separate human review of
``eval/parsing_quality_sample.csv`` for this run. For each of the 50
samples, this script rasterizes the PDF pages around the parser's recorded
page range (one page before ``page_start``, one page after ``page_end``,
clamped to the document's real page count -- exactly the window needed to
catch the boundary bugs [M1-04c] targets), sends them to the model
alongside the parser's own claim (title, predicted type, text excerpt),
and asks it to independently judge boundary correctness, clause type and
provenance plausibility -- the same three judgments [M1-08]'s human
protocol made.

The model's judgment is written directly into ``eval/parsing_quality_
sample.csv``'s seven judgment columns (``boundary_correct``,
``boundary_notes``, ``reference_clause_type``, ``provenance_correct``,
``provenance_notes``, ``failure_mode_tag``, ``reviewer_notes``) -- this IS
the recorded validation for [M1-08b], not an advisory aid sitting
alongside it. ``failure_mode_tag`` is derived here (see
[derive_failure_mode_tag]) from which of the three judgments failed first,
rather than asked of the model directly, since the model already explains
itself in ``boundary_notes``/``provenance_notes``/``reviewer_notes`` and a
short derived tag keeps ``scripts/score_parsing_quality.py``'s failure-mode
frequency table meaningful without inventing a second free-form
vocabulary. Every per-sample judgment is also kept in
``eval/temp/sample_validations/validation_sample_{id:03d}.json`` (the raw
model output, ``llm_``-prefixed) as the audit trail behind the CSV values.

Both stages are cached and resumable:

- Rasterized pages: ``eval/temp/sample_imgs/sample_{id:03d}/page_{n:03d}.png``
  -- skipped entirely if already on disk.
- Per-sample validations: ``eval/temp/sample_validations/
  validation_sample_{id:03d}.json`` -- skipped (loaded from disk) if
  already written, so a kill/crash mid-run loses at most the in-flight
  sample, and a rerun after the CSV schema/mapping changes here does not
  re-spend the LLM call.

Processing is strictly sequential across the 50 samples (no thread pool) --
this is a slow, careful validation pass, not a throughput-critical
pipeline stage. If the LLM call for a sample fails, it is retried after a
5 second sleep, up to 3 attempts total; unlike ``scripts/build_corpus.py``'s
classifier, there is no sane fallback value for a validation judgment, so
exhausting retries is a hard failure, not a silent default.

Run via ``make validate-parsing-quality-sample``, after
``make sample-parsing-quality`` has produced a fresh
``eval/parsing_quality_sample.csv``. Follow with
``make score-parsing-quality`` directly -- no manual annotation step sits
between them for this project.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import cast

import fitz
from langchain_core.messages import HumanMessage
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field
from tqdm import tqdm

from domain.clause_classification import ClauseType
from infrastructure.config.llm_client_factory import build_chat_model
from infrastructure.config.settings import LlmSettings, get_llm_settings
from infrastructure.parsing.manifest import read_manifest

SAMPLE_CSV_PATH = Path("eval/parsing_quality_sample.csv")
MANIFEST_PATH = Path("data/policies/manifest.csv")
RAW_DIR = Path("data/policies/raw")
IMAGE_CACHE_DIR = Path("eval/temp/sample_imgs")
VALIDATION_OUTPUT_DIR = Path("eval/temp/sample_validations")

RASTER_DPI = 150

VALIDATION_MODEL = "google/gemini-3.7-flash"
VALIDATION_PROVIDER_ORDER = ["google-vertex"]
VALIDATION_ALLOW_FALLBACKS = False
VALIDATION_MAX_ATTEMPTS = 3
VALIDATION_RETRY_DELAY_SECONDS = 5.0


class LLMValidationOutput(BaseModel):
    """The model's judgment for one sampled clause.

    The recorded [M1-08b] validation, mapped into ``eval/
    parsing_quality_sample.csv``'s judgment columns by
    [apply_llm_validation]. Field names stay ``llm_``-prefixed here (and in
    the raw JSON audit trail under ``eval/temp/sample_validations/``) to
    keep this model's own output distinct from the CSV's plain column
    names it gets written into.
    """

    llm_boundary_correct: bool = Field(
        ...,
        description=(
            "Whether the clause, on the page range the parser recorded, "
            "starts and ends where it should in the attached pages -- not "
            "merged with a neighboring clause, not cut off mid-content."
        ),
    )
    llm_boundary_notes: str = Field(
        ..., description="Short note explaining the boundary judgment."
    )
    llm_reference_clause_type: ClauseType = Field(
        ...,
        description=(
            "Your own classification of the clause, judged independently "
            "of what the parser predicted."
        ),
    )
    llm_provenance_correct: bool = Field(
        ...,
        description=(
            "Whether the attached pages' visible content looks consistent "
            "with the stated insurer/product line -- a coarse sanity "
            "check, not a definitive audit."
        ),
    )
    llm_provenance_notes: str = Field(
        ..., description="Short note explaining the provenance judgment."
    )
    llm_reasoning: str = Field(
        ..., description="Short overall rationale tying the three judgments together."
    )


def resolve_page_range(
    page_start: int, page_end: int, page_count: int
) -> tuple[int, int]:
    """Return the (first, last) page window to rasterize, one page of margin.

    Clamped to the document's real page count so an out-of-range page is
    never requested.
    """
    first = max(1, page_start - 1)
    last = min(page_count, page_end + 1)
    return first, last


def rasterize_pages(
    pdf_path: Path,
    sample_id: int,
    first_page: int,
    last_page: int,
    dpi: int = RASTER_DPI,
) -> list[Path]:
    """Rasterize ``pdf_path``'s pages in ``[first_page, last_page]`` to PNG.

    Idempotent: skips opening the PDF entirely if every target PNG already
    exists on disk. Reuses the exact PyMuPDF rasterization pattern already
    used for OCR (see [infrastructure.parsing.ocr.TesseractOcrExtractor]).
    """
    sample_dir = IMAGE_CACHE_DIR / f"sample_{sample_id:03d}"
    targets = {
        page_num: sample_dir / f"page_{page_num:03d}.png"
        for page_num in range(first_page, last_page + 1)
    }
    if all(path.exists() for path in targets.values()):
        return [targets[page_num] for page_num in sorted(targets)]

    sample_dir.mkdir(parents=True, exist_ok=True)
    document = fitz.open(pdf_path)
    try:
        zoom = dpi / 72
        matrix = fitz.Matrix(zoom, zoom)
        for page_num, png_path in targets.items():
            if png_path.exists():
                continue
            page = document.load_page(page_num - 1)
            pixmap = page.get_pixmap(matrix=matrix)
            pixmap.save(png_path)
    finally:
        document.close()
    return [targets[page_num] for page_num in sorted(targets)]


def build_validation_message(
    sample_row: dict[str, str], image_paths: list[Path]
) -> list[str | dict[str, object]]:
    """Build the multimodal human-turn content: the parser's claim plus pages."""
    prompt_text = (
        "You are validating the output of an automated Brazilian insurance "
        "policy clause parser against the attached page images. The pages "
        "are a window around the parser's recorded boundary: one page "
        "before the recorded start and one page after the recorded end, so "
        "you can judge whether the boundary is correct, truncated, or "
        "merged with a neighboring clause.\n\n"
        "Parser's claim:\n"
        f"- Document: {sample_row['filename']} "
        f"(document_id={sample_row['document_id']})\n"
        f"- Product line: {sample_row['product_line']}\n"
        f"- Insurer: {sample_row['insurer']}\n"
        f"- Clause title: {sample_row['title']}\n"
        f"- Predicted clause type: {sample_row['predicted_clause_type']}\n"
        f"- Recorded page range: {sample_row['page_start']}-{sample_row['page_end']}\n"
        f"- Text excerpt captured by the parser:\n{sample_row['text_excerpt']}\n\n"
        "IMPORTANT -- the parser produces a clause TREE, not a flat list. "
        "A clause's own numbered sub-clauses (e.g. 3.4, 3.4.1 under clause "
        "3) are captured as SEPARATE records with their own page ranges, "
        "not as part of this record. So if the content continuing past the "
        "recorded page range is itself a numbered sub-clause of this "
        "clause, that content is NOT missing -- it lives in its own record "
        "-- and this record is NOT truncated. Judge only whether THIS "
        "record's own captured text (the section's own heading and any "
        "prose, lettered items or lists belonging directly to it, before "
        "its first numbered sub-clause) starts and ends where it should.\n\n"
        "Judge, independently of the parser's claim:\n"
        "1. Boundary correctness -- does this record's own content actually "
        "start and end on the recorded pages, or is it truncated "
        "(cutting off prose or lettered items that belong directly to it), "
        "merged with a neighboring clause's content, or misattributed to "
        "the wrong pages?\n"
        "2. The clause's true type (coverage / exclusion / condition / "
        "definition / procedure / other), from what the pages actually "
        "show.\n"
        "3. Provenance plausibility -- do the attached pages look "
        "consistent with the stated insurer and product line?\n"
        "Provide a short rationale tying these three judgments together."
    )
    content: list[str | dict[str, object]] = [{"type": "text", "text": prompt_text}]
    for png_path in image_paths:
        encoded = base64.b64encode(png_path.read_bytes()).decode("ascii")
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{encoded}"},
            }
        )
    return content


def build_validation_chain(
    llm_settings: LlmSettings,
) -> Runnable[list[HumanMessage], LLMValidationOutput]:
    """Build the structured-output chain for the image-validation model."""
    llm = build_chat_model(
        llm_settings,
        VALIDATION_MODEL,
        provider_order=VALIDATION_PROVIDER_ORDER,
        allow_fallbacks=VALIDATION_ALLOW_FALLBACKS,
    )
    return cast(
        Runnable[list[HumanMessage], LLMValidationOutput],
        llm.with_structured_output(LLMValidationOutput),
    )


def call_llm_with_retry(
    chain: Runnable[list[HumanMessage], LLMValidationOutput],
    content: list[str | dict[str, object]],
) -> LLMValidationOutput:
    """Invoke the validation chain, retrying transient failures.

    Retries up to [VALIDATION_MAX_ATTEMPTS] times, sleeping
    [VALIDATION_RETRY_DELAY_SECONDS] between attempts. Unlike the parse
    pipeline's classifier, this re-raises on final failure -- there is no
    sane fallback value for a validation judgment.
    """
    last_exc: Exception | None = None
    for attempt in range(1, VALIDATION_MAX_ATTEMPTS + 1):
        try:
            return chain.invoke([HumanMessage(content=content)])
        except Exception as exc:
            last_exc = exc
            if attempt < VALIDATION_MAX_ATTEMPTS:
                time.sleep(VALIDATION_RETRY_DELAY_SECONDS)
    assert last_exc is not None
    raise last_exc


CLAIM_FINGERPRINT_FIELDS = (
    "clause_id",
    "document_id",
    "page_start",
    "page_end",
    "predicted_clause_type",
    "title",
    "text_excerpt",
)


def claim_fingerprint(sample_row: dict[str, str]) -> str:
    """Content hash of everything the model is shown about one clause.

    [M1-08c]: resumability used to key only on ``sample_id`` -- the row's
    position in the CSV -- so re-running against a corpus whose clause
    boundaries had changed silently reused the *previous* measurement's
    judgments instead of re-validating. It surfaced only because the run
    "validated" all 50 samples in under a second instead of ~8 minutes.
    Keying on the claim itself turns a stale file into a cache miss, the
    same content-addressed pattern
    [infrastructure.parsing.boundary_escalation_cache] already uses.
    """
    payload = "|".join(sample_row[field] for field in CLAIM_FINGERPRINT_FIELDS)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_sample(
    sample_row: dict[str, str],
    page_counts: dict[str, int],
    filenames: dict[str, str],
    chain: Runnable[list[HumanMessage], LLMValidationOutput],
) -> dict[str, object]:
    """Validate one sample, resuming from disk only for the identical claim."""
    sample_id = int(sample_row["sample_id"])
    validation_path = VALIDATION_OUTPUT_DIR / f"validation_sample_{sample_id:03d}.json"
    fingerprint = claim_fingerprint(sample_row)
    if validation_path.exists():
        cached = cast(
            dict[str, object], json.loads(validation_path.read_text(encoding="utf-8"))
        )
        if cached.get("claim_fingerprint") == fingerprint:
            return cached

    document_id = sample_row["document_id"]
    page_start = int(sample_row["page_start"])
    page_end = int(sample_row["page_end"])
    first_page, last_page = resolve_page_range(
        page_start, page_end, page_counts[document_id]
    )

    pdf_path = RAW_DIR / filenames[document_id]
    image_paths = rasterize_pages(pdf_path, sample_id, first_page, last_page)

    content = build_validation_message(sample_row, image_paths)
    output = call_llm_with_retry(chain, content)

    result: dict[str, object] = {
        "sample_id": sample_id,
        "clause_id": sample_row["clause_id"],
        "document_id": document_id,
        "claim_fingerprint": fingerprint,
        **output.model_dump(mode="json"),
    }
    VALIDATION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    validation_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def derive_failure_mode_tag(
    *, boundary_correct: bool, type_correct: bool, provenance_correct: bool
) -> str:
    """Short tag naming the first failing judgment; empty when all three pass.

    Priority order: boundary first, since a wrong boundary usually explains
    a type mismatch too (the model is judging text the parser never
    actually captured for this clause), then type, then provenance.
    """
    if not boundary_correct:
        return "boundary_mismatch_llm"
    if not type_correct:
        return "type_mismatch_llm"
    if not provenance_correct:
        return "provenance_mismatch_llm"
    return ""


def apply_llm_validation(
    row: dict[str, str], validation: dict[str, object]
) -> dict[str, str]:
    """Fill ``row``'s seven judgment columns from the model's validation.

    This is the recorded [M1-08b] validation, not an advisory overlay --
    see the module docstring.
    """
    boundary_correct = bool(validation["llm_boundary_correct"])
    reference_clause_type = str(validation["llm_reference_clause_type"])
    provenance_correct = bool(validation["llm_provenance_correct"])
    type_correct = reference_clause_type == row["predicted_clause_type"]

    updated = dict(row)
    updated["boundary_correct"] = "TRUE" if boundary_correct else "FALSE"
    updated["boundary_notes"] = str(validation["llm_boundary_notes"])
    updated["reference_clause_type"] = reference_clause_type
    updated["provenance_correct"] = "TRUE" if provenance_correct else "FALSE"
    updated["provenance_notes"] = str(validation["llm_provenance_notes"])
    updated["failure_mode_tag"] = derive_failure_mode_tag(
        boundary_correct=boundary_correct,
        type_correct=type_correct,
        provenance_correct=provenance_correct,
    )
    updated["reviewer_notes"] = str(validation["llm_reasoning"])
    return updated


def main() -> None:
    """Run the LLM validation pass and write judgments into the sample CSV."""
    if not SAMPLE_CSV_PATH.exists():
        raise FileNotFoundError(
            f"{SAMPLE_CSV_PATH} does not exist. "
            "Run `make sample-parsing-quality` first."
        )

    with SAMPLE_CSV_PATH.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if fieldnames is None:
        raise ValueError(f"{SAMPLE_CSV_PATH} has no header row.")

    manifest_records = read_manifest(MANIFEST_PATH)
    page_counts = {
        record["id"]: int(record["page_count"]) for record in manifest_records
    }
    filenames = {record["id"]: record["filename"] for record in manifest_records}

    llm_settings = get_llm_settings()
    chain = build_validation_chain(llm_settings)

    updated_rows: list[dict[str, str]] = []
    for row in tqdm(rows, desc="Validating samples", unit="sample"):
        try:
            validation = validate_sample(row, page_counts, filenames, chain)
        except Exception:
            print(f"FAILED on sample_id={row['sample_id']}", file=sys.stderr)
            raise
        updated_rows.append(apply_llm_validation(row, validation))

    with SAMPLE_CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(updated_rows)

    print(f"Wrote {len(rows)} validations to {VALIDATION_OUTPUT_DIR}")
    print(f"Filled judgment columns for {len(rows)} rows in {SAMPLE_CSV_PATH}")


if __name__ == "__main__":
    main()
