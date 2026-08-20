#!/usr/bin/env python3
"""[M1-04d]: run the vision-LLM boundary-escalation pass over the corpus.

Opt-in, deliberately **not** part of `make parse` -- the DoD requires
measuring this pass's actual corpus-wide cost (call count, estimated $
cost, wall-clock time) before deciding whether it should run by default.
This script measures and reports that cost (written to
``eval/boundary_escalation_cost_report.json``); the decision to fold it
into `make parse` by default is left to a human, informed by that report.

For each document, loads the clause tree [scripts.build_clause_tree]
already segmented and cached under ``data/cache/clause_trees/`` (never
re-segments -- a missing cache means that script hasn't run yet, and
silently re-deriving it here would hide that instead of failing loudly),
flags suspicious clauses (see [application.use_cases.boundary_escalation.
find_suspicious_clauses]), runs each through a vision-capable model, and
**overwrites the same cache path** with the (possibly boundary-corrected)
tree -- so the next `make parse`/`build-clause-tree` picks it up
transparently (``build_clause_tree.py``'s own cache-write is a no-op once
the file exists, so it won't clobber this).

Run via `make escalate-vision-boundaries`, after `make build-clause-tree`.
Use ``--document-ids`` to smoke-test a small subset before a full run.
"""

import argparse
import json
from pathlib import Path

from application.use_cases.boundary_escalation import escalate_boundaries
from application.use_cases.clause_segmentation import CLAUSE_SEGMENTATION_VERSION
from domain.clause_tree import ClauseTree
from infrastructure.config.llm_client_factory import build_chat_model
from infrastructure.config.settings import get_llm_settings, get_parsing_settings
from infrastructure.parsing.boundary_escalation_cache import (
    CachingBoundaryVisionReviewer,
)
from infrastructure.parsing.boundary_vision import (
    LangchainBoundaryVisionReviewer,
    PyMuPdfPageRasterizer,
)
from infrastructure.parsing.clause_tree_caching import (
    clause_tree_cache_path,
    compute_clause_tree_cache_key,
    read_clause_tree_cache,
    write_clause_tree_cache,
)
from infrastructure.parsing.manifest import read_manifest

MANIFEST_PATH = Path("data/policies/manifest.csv")
RAW_DIR = Path("data/policies/raw")
BOUNDARY_ESCALATION_CACHE_PATH = Path("data/cache/boundary_escalation/cache.jsonl")
COST_REPORT_PATH = Path("eval/boundary_escalation_cost_report.json")

# Matches scripts/validate_parsing_quality_sample.py's existing pin for the
# same model -- reused rather than reinvented, per the DoD's "no new LLM
# client integration" ask. Fallback disabled so a transient provider outage
# surfaces as an exception (caught and retried by _review_with_retry)
# instead of silently rerouting to a different, unvalidated upstream.
VISION_MODEL_PROVIDER_ORDER = ["google-vertex"]
VISION_ALLOW_FALLBACKS = False

# Rough, provider-published per-1M-token prices at time of writing -- check
# against the provider's current pricing before treating estimated_cost_usd
# as authoritative for a real budgeting decision.
VISION_INPUT_COST_PER_1M_TOKENS_USD = 0.30
VISION_OUTPUT_COST_PER_1M_TOKENS_USD = 2.50


def load_clause_tree(document_id: str, filename: str) -> tuple[Path, ClauseTree]:
    """Load a document's clause tree from its [M1-04] cache, plus its cache path.

    Raises ``FileNotFoundError`` if that cache is missing -- run
    ``scripts/build_clause_tree.py`` first.
    """
    cache_key = compute_clause_tree_cache_key(CLAUSE_SEGMENTATION_VERSION)
    path = clause_tree_cache_path(document_id, cache_key)
    if not path.exists():
        raise FileNotFoundError(
            f"No clause-tree cache for {filename} (document {document_id}) "
            f"at {path}. Run `scripts/build_clause_tree.py` first."
        )
    return path, read_clause_tree_cache(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--document-ids",
        type=str,
        default=None,
        help=(
            "Comma-separated manifest document ids to run (default: the whole corpus)."
        ),
    )
    parser.add_argument(
        "--continue-on-failure",
        action="store_true",
        help=(
            "Process every document even if one fails (e.g. exhausted "
            "retries), raising a single aggregate error at the end instead "
            "of failing fast."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Run the vision-escalation pass and write the cost report."""
    args = _parse_args()
    settings = get_llm_settings()
    if settings.llm_model_vision is None:
        raise ValueError(
            "LLM_MODEL_VISION is not set. Set it in .env before running "
            "`make escalate-vision-boundaries`."
        )

    llm = build_chat_model(
        settings,
        settings.llm_model_vision,
        provider_order=VISION_MODEL_PROVIDER_ORDER,
        allow_fallbacks=VISION_ALLOW_FALLBACKS,
    )
    inner_reviewer = LangchainBoundaryVisionReviewer(llm)
    reviewer = CachingBoundaryVisionReviewer(
        inner_reviewer,
        model=settings.llm_model_vision,
        cache_path=BOUNDARY_ESCALATION_CACHE_PATH,
    )
    parsing_settings = get_parsing_settings()
    rasterizer = PyMuPdfPageRasterizer(dpi=parsing_settings.ocr_dpi)

    manifest_records = read_manifest(MANIFEST_PATH)
    if args.document_ids:
        wanted = {doc_id.strip() for doc_id in args.document_ids.split(",")}
        manifest_records = [r for r in manifest_records if r["id"] in wanted]
        missing = wanted - {r["id"] for r in manifest_records}
        if missing:
            raise ValueError(f"Document id(s) not found in manifest: {sorted(missing)}")

    documents_processed = 0
    suspicious_clause_count = 0
    applied_count = 0
    split_suggested_count = 0
    failures: list[Exception] = []

    for entry in manifest_records:
        document_id = entry["id"]
        filename = entry["filename"]
        page_count = int(entry["page_count"])
        pdf_path = RAW_DIR / filename
        try:
            cache_path, tree = load_clause_tree(document_id, filename)
            revised_tree, outcomes = escalate_boundaries(
                tree,
                pdf_path,
                reviewer=reviewer,
                rasterizer=rasterizer,
                page_count=page_count,
                max_page_span=parsing_settings.clause_max_page_span,
                max_char_count=parsing_settings.clause_max_char_count,
            )
        except Exception as exc:  # noqa: BLE001 -- aggregated below
            if args.continue_on_failure:
                failures.append(exc)
                continue
            raise

        documents_processed += 1
        suspicious_clause_count += len(outcomes)
        applied_count += sum(1 for outcome in outcomes if outcome.applied)
        split_suggested_count += sum(
            1 for outcome in outcomes if outcome.review.split_suggested
        )
        if outcomes:
            write_clause_tree_cache(revised_tree, cache_path)
        print(
            f"{filename}: suspicious={len(outcomes)} "
            f"applied={sum(1 for o in outcomes if o.applied)}"
        )

    stats = inner_reviewer.stats
    estimated_cost_usd = (
        stats.total_input_tokens / 1_000_000 * VISION_INPUT_COST_PER_1M_TOKENS_USD
        + stats.total_output_tokens / 1_000_000 * VISION_OUTPUT_COST_PER_1M_TOKENS_USD
    )
    report = {
        "documents_processed": documents_processed,
        "suspicious_clause_count": suspicious_clause_count,
        "applied_count": applied_count,
        "split_suggested_count": split_suggested_count,
        "cache_hit_count": suspicious_clause_count - stats.call_count,
        "call_count": stats.call_count,
        "total_input_tokens": stats.total_input_tokens,
        "total_output_tokens": stats.total_output_tokens,
        "wall_clock_seconds": stats.total_seconds,
        "estimated_cost_usd": round(estimated_cost_usd, 4),
        "pricing_note": (
            "estimated_cost_usd uses hardcoded per-1M-token price constants "
            "in this script -- check against the provider's current pricing "
            "before treating it as authoritative."
        ),
    }
    COST_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    COST_REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))

    if failures:
        raise ExceptionGroup(f"{len(failures)} document(s) failed", failures)


if __name__ == "__main__":
    main()
