"""Parquet cache for clause trees.

Reuses the same flat-row-per-node convention as [infrastructure.parsing.
caching] and [infrastructure.parsing.boilerplate_caching]: one row per
[domain.clause_tree.Clause], document-level fields (including the JSON-
encoded warning list, which has no natural per-clause row) carried as table
metadata. Keyed by document id and a hash of the segmentation algorithm
version alone -- the upstream boilerplate-removal version is already baked
into the input artifact one hop back (``data/cache/boilerplate_removed/``),
so it doesn't need to be folded in again here.
"""

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from domain.clause_tree import (
    Clause,
    ClauseTree,
    ClauseTreeReport,
    ClauseTreeWarning,
    HeadingConvention,
)

CLAUSE_TREE_CACHE_DIR = Path("data/cache/clause_trees")

_ROW_SCHEMA = pa.schema(
    [
        ("clause_id", pa.string()),
        ("path", pa.string()),
        ("numbering_label", pa.string()),
        ("title", pa.string()),
        ("convention", pa.string()),
        ("depth", pa.int64()),
        ("parent_id", pa.string()),
        ("child_ids", pa.string()),
        ("content_lines", pa.string()),
        ("page_start", pa.int64()),
        ("page_end", pa.int64()),
        ("bundle_section", pa.string()),
        ("bundle_confidence", pa.string()),
        ("is_depth_anomaly", pa.bool_()),
    ]
)


def compute_clause_tree_cache_key(clause_segmentation_version: str) -> str:
    """Hash the input that should invalidate a cached clause tree."""
    payload = clause_segmentation_version.encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def clause_tree_cache_path(document_id: str, cache_key: str) -> Path:
    """Return the Parquet cache path for a document/cache-key pair."""
    return CLAUSE_TREE_CACHE_DIR / f"{document_id}__{cache_key}.parquet"


def write_clause_tree_cache(tree: ClauseTree, path: Path) -> None:
    """Write a clause tree to Parquet as flat, per-clause rows."""
    rows = [
        {
            "clause_id": clause.clause_id,
            "path": clause.path,
            "numbering_label": clause.numbering_label,
            "title": clause.title,
            "convention": clause.convention.value,
            "depth": clause.depth,
            "parent_id": clause.parent_id or "",
            "child_ids": ",".join(clause.child_ids),
            "content_lines": "\n".join(clause.content_lines),
            "page_start": clause.page_start,
            "page_end": clause.page_end,
            "bundle_section": clause.bundle_section or "",
            "bundle_confidence": clause.bundle_confidence or "",
            "is_depth_anomaly": clause.is_depth_anomaly,
        }
        for clause in tree.all_clauses
    ]
    table = pa.Table.from_pylist(rows, schema=_ROW_SCHEMA)

    warning_payload = [
        {
            "page_number": warning.page_number,
            "kind": warning.kind,
            "detail": warning.detail,
        }
        for warning in tree.report.warnings
    ]
    table = table.replace_schema_metadata(
        {
            "document_id": tree.document_id,
            "filename": tree.filename,
            "orphan_char_count": str(tree.report.orphan_char_count),
            "total_char_count": str(tree.report.total_char_count),
            "orphan_ratio": repr(tree.report.orphan_ratio),
            "extraction_mode": tree.report.extraction_mode,
            "warnings_json": json.dumps(warning_payload),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def read_clause_tree_cache(path: Path) -> ClauseTree:
    """Read a cached clause tree back from Parquet."""
    table = pq.read_table(path)
    metadata = table.schema.metadata or {}
    document_id = metadata[b"document_id"].decode()
    filename = metadata[b"filename"].decode()
    extraction_mode = metadata[b"extraction_mode"].decode()
    orphan_char_count = int(metadata[b"orphan_char_count"].decode())
    total_char_count = int(metadata[b"total_char_count"].decode())
    orphan_ratio = float(metadata[b"orphan_ratio"].decode())
    warnings = tuple(
        ClauseTreeWarning(
            document_id=document_id,
            page_number=row["page_number"],
            kind=row["kind"],
            detail=row["detail"],
        )
        for row in json.loads(metadata[b"warnings_json"].decode())
    )

    clauses = tuple(
        Clause(
            document_id=document_id,
            clause_id=row["clause_id"],
            path=row["path"],
            numbering_label=row["numbering_label"],
            title=row["title"],
            convention=HeadingConvention(row["convention"]),
            depth=row["depth"],
            parent_id=row["parent_id"] or None,
            child_ids=tuple(row["child_ids"].split(",")) if row["child_ids"] else (),
            content_lines=(
                tuple(row["content_lines"].split("\n")) if row["content_lines"] else ()
            ),
            page_start=row["page_start"],
            page_end=row["page_end"],
            bundle_section=row["bundle_section"] or None,
            bundle_confidence=row["bundle_confidence"] or None,
            is_depth_anomaly=row["is_depth_anomaly"],
        )
        for row in table.to_pylist()
    )
    roots = tuple(clause for clause in clauses if clause.parent_id is None)

    report = ClauseTreeReport(
        document_id=document_id,
        filename=filename,
        clause_count=len(clauses),
        max_depth=max((clause.depth for clause in clauses), default=0),
        orphan_char_count=orphan_char_count,
        total_char_count=total_char_count,
        orphan_ratio=orphan_ratio,
        extraction_mode=extraction_mode,
        warnings=warnings,
    )
    return ClauseTree(
        document_id=document_id,
        filename=filename,
        roots=roots,
        all_clauses=clauses,
        report=report,
    )
