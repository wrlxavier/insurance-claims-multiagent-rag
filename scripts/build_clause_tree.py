#!/usr/bin/env python3
"""Recover a parent-referenced clause tree from heading numbering.

Runs over every already-boilerplate-cleaned policy document. Post-processing
stage for [M1-04], running after [M1-03] and before clause
type classification [M1-05]. Reads each document from
``data/cache/boilerplate_removed/`` -- this script never re-runs boilerplate
removal itself, since a missing cache means ``scripts/remove_boilerplate.py``
has not been run yet, and silently re-deriving it here would hide that
instead of failing loudly. Segments the document into a clause hierarchy,
caches the result under ``data/cache/clause_trees/``, and writes a
per-document report to ``docs/CLAUSE_TREE_REPORT.md``.

Documents whose orphan-text ratio exceeds
``CLAUSE_TREE_ORPHAN_RATIO_THRESHOLD`` raise
[domain.clause_tree.OrphanTextExceedsThresholdError] rather than silently
emitting a broken tree -- pass ``--continue-on-orphan-failure`` to keep
processing the rest of the corpus and raise a single aggregate error at the
end instead of failing fast on the first offending document.

Use ``--checkpoint-docs`` to run a small subset and write only their
human-reviewable outlines under ``docs/clause_tree_checkpoints/``, without
touching the aggregate report or requiring the full corpus.
"""

import argparse
from pathlib import Path

from application.use_cases.boilerplate_removal import BOILERPLATE_REMOVAL_VERSION
from application.use_cases.clause_segmentation import (
    CLAUSE_SEGMENTATION_VERSION,
    segment_document,
)
from domain.clause_tree import ClauseTree, OrphanTextExceedsThresholdError
from domain.extracted_text import ExtractedDocument
from infrastructure.config.settings import get_parsing_settings
from infrastructure.parsing.boilerplate_caching import (
    boilerplate_cache_path,
    compute_boilerplate_cache_key,
)
from infrastructure.parsing.caching import read_cache
from infrastructure.parsing.clause_tree_caching import (
    clause_tree_cache_path,
    compute_clause_tree_cache_key,
    write_clause_tree_cache,
)
from infrastructure.parsing.clause_tree_outline import render_outline
from infrastructure.parsing.clause_tree_report import ClauseTreeReportRow, render_report
from infrastructure.parsing.manifest import read_manifest

MANIFEST_PATH = Path("data/policies/manifest.csv")
REPORT_PATH = Path("docs/CLAUSE_TREE_REPORT.md")
CHECKPOINT_DIR = Path("docs/clause_tree_checkpoints")


def load_boilerplate_removed_document(
    document_id: str, filename: str
) -> ExtractedDocument:
    """Load a document from its upstream [M1-03] cache.

    Raises ``FileNotFoundError`` if that cache is missing -- run
    ``scripts/remove_boilerplate.py`` first.
    """
    cache_key = compute_boilerplate_cache_key(BOILERPLATE_REMOVAL_VERSION)
    path = boilerplate_cache_path(document_id, cache_key)
    if not path.exists():
        raise FileNotFoundError(
            f"No boilerplate-removed cache for {filename} (document "
            f"{document_id}) at {path}. Run `scripts/remove_boilerplate.py` "
            "first."
        )
    return read_cache(path)


def segment_and_cache(document: ExtractedDocument) -> tuple[ClauseTree, bool]:
    """Segment and cache the result. Returns (tree, was_cached)."""
    tree = segment_document(document)
    cache_key = compute_clause_tree_cache_key(CLAUSE_SEGMENTATION_VERSION)
    path = clause_tree_cache_path(document.document_id, cache_key)
    was_cached = path.exists()
    if not was_cached:
        write_clause_tree_cache(tree, path)
    return tree, was_cached


def _build_report_row(
    document_id: str,
    filename: str,
    page_count: int,
    tree: ClauseTree,
    threshold: float,
) -> ClauseTreeReportRow:
    report = tree.report
    exceeds = report.orphan_ratio > threshold
    print(
        f"{filename}: clauses={report.clause_count} max_depth={report.max_depth} "
        f"orphan_ratio={report.orphan_ratio:.3f} mode={report.extraction_mode} "
        f"warnings={len(report.warnings)}" + (" [EXCEEDS THRESHOLD]" if exceeds else "")
    )
    return ClauseTreeReportRow(
        document_id=document_id,
        filename=filename,
        page_count=page_count,
        clause_count=report.clause_count,
        max_depth=report.max_depth,
        orphan_ratio=report.orphan_ratio,
        extraction_mode=report.extraction_mode,
        warning_count=len(report.warnings),
        exceeds_threshold=exceeds,
    )


def run_clause_segmentation(
    *, threshold: float, continue_on_orphan_failure: bool
) -> tuple[list[ClauseTreeReportRow], list[OrphanTextExceedsThresholdError]]:
    """Process every manifest row. Returns (rows, failures).

    ``failures`` is only ever non-empty when ``continue_on_orphan_failure``
    is set -- otherwise the first threshold breach raises immediately. The
    caller is responsible for writing the report before acting on
    ``failures``, so a persisted run always reflects every document that
    was actually processed, not just the ones before the first failure.
    """
    rows: list[ClauseTreeReportRow] = []
    failures: list[OrphanTextExceedsThresholdError] = []
    for entry in read_manifest(MANIFEST_PATH):
        document_id = entry["id"]
        filename = entry["filename"]
        document = load_boilerplate_removed_document(document_id, filename)
        tree, _was_cached = segment_and_cache(document)
        rows.append(
            _build_report_row(
                document_id, filename, int(entry["page_count"]), tree, threshold
            )
        )
        if tree.report.orphan_ratio > threshold:
            error = OrphanTextExceedsThresholdError(
                document_id=document_id,
                filename=filename,
                orphan_ratio=tree.report.orphan_ratio,
                threshold=threshold,
                clause_count=tree.report.clause_count,
            )
            if continue_on_orphan_failure:
                failures.append(error)
            else:
                raise error

    return rows, failures


def run_checkpoint(document_ids: set[str]) -> None:
    """Segment a small subset and write only their human-reviewable outlines."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    found: set[str] = set()
    for entry in read_manifest(MANIFEST_PATH):
        document_id = entry["id"]
        if document_id not in document_ids:
            continue
        found.add(document_id)
        filename = entry["filename"]
        document = load_boilerplate_removed_document(document_id, filename)
        tree = segment_document(document)
        stem = Path(filename).stem
        outline_path = CHECKPOINT_DIR / f"{document_id}_{stem}.md"
        outline_path.write_text(render_outline(tree), encoding="utf-8")
        print(
            f"{filename}: clauses={tree.report.clause_count} "
            f"max_depth={tree.report.max_depth} "
            f"orphan_ratio={tree.report.orphan_ratio:.3f} -> {outline_path}"
        )
    missing = document_ids - found
    if missing:
        raise ValueError(f"Document id(s) not found in manifest: {sorted(missing)}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-docs",
        type=str,
        default=None,
        help=(
            "Comma-separated manifest document ids to run in checkpoint-only "
            "mode (writes outlines under docs/clause_tree_checkpoints/, skips "
            "the aggregate report)."
        ),
    )
    parser.add_argument(
        "--continue-on-orphan-failure",
        action="store_true",
        help=(
            "Process every document even if one exceeds the orphan-ratio "
            "threshold, raising a single aggregate error at the end instead "
            "of failing fast."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Run clause-tree segmentation over the corpus (or a checkpoint subset)."""
    args = _parse_args()
    settings = get_parsing_settings()

    if args.checkpoint_docs:
        document_ids = {doc_id.strip() for doc_id in args.checkpoint_docs.split(",")}
        run_checkpoint(document_ids)
        return

    rows, failures = run_clause_segmentation(
        threshold=settings.clause_tree_orphan_ratio_threshold,
        continue_on_orphan_failure=args.continue_on_orphan_failure,
    )
    REPORT_PATH.write_text(
        render_report(rows, threshold=settings.clause_tree_orphan_ratio_threshold),
        encoding="utf-8",
    )
    if failures:
        raise ExceptionGroup(
            f"{len(failures)} document(s) exceeded the orphan-ratio threshold", failures
        )


if __name__ == "__main__":
    main()
