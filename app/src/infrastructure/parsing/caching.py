"""Parquet cache for extracted documents.

Keyed by document id and a hash of (extractor version, normalization
version), so unchanged inputs skip re-extraction on the next run.
"""

import hashlib
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from domain.extracted_text import ExtractedDocument, ExtractedPage, ExtractedSpan

CACHE_DIR = Path("data/cache/extraction")


def compute_cache_key(extractor_version: str, normalization_version: str) -> str:
    """Hash the inputs that should invalidate a cached extraction."""
    payload = f"{extractor_version}|{normalization_version}".encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def cache_path(document_id: str, cache_key: str) -> Path:
    """Return the Parquet cache path for a document/cache-key pair."""
    return CACHE_DIR / f"{document_id}__{cache_key}.parquet"


def write_cache(document: ExtractedDocument, path: Path) -> None:
    """Write an extracted document to Parquet as flat, per-span rows."""
    rows = [
        {
            "page_number": span.page_number,
            "line_id": span.line_id,
            "order": span.order,
            "bbox_x0": span.bbox[0],
            "bbox_y0": span.bbox[1],
            "bbox_x1": span.bbox[2],
            "bbox_y1": span.bbox[3],
            "font_size": span.font_size,
            "font_name": span.font_name,
            "text": span.text,
        }
        for page in document.pages
        for span in page.spans
    ]
    table = pa.Table.from_pylist(rows)
    table = table.replace_schema_metadata(
        {
            "document_id": document.document_id,
            "filename": document.filename,
            "extractor_version": document.extractor_version,
            "page_count": str(len(document.pages)),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def read_cache(path: Path) -> ExtractedDocument:
    """Read a cached extracted document back from Parquet."""
    table = pq.read_table(path)
    metadata = table.schema.metadata or {}
    document_id = metadata[b"document_id"].decode()
    filename = metadata[b"filename"].decode()
    extractor_version = metadata[b"extractor_version"].decode()
    page_count = int(metadata[b"page_count"].decode())

    spans_by_page: dict[int, list[ExtractedSpan]] = {}
    for row in table.to_pylist():
        span = ExtractedSpan(
            document_id=document_id,
            page_number=row["page_number"],
            line_id=row["line_id"],
            order=row["order"],
            bbox=(row["bbox_x0"], row["bbox_y0"], row["bbox_x1"], row["bbox_y1"]),
            font_size=row["font_size"],
            font_name=row["font_name"],
            text=row["text"],
        )
        spans_by_page.setdefault(span.page_number, []).append(span)

    pages: list[ExtractedPage] = []
    for page_number in range(1, page_count + 1):
        spans = tuple(spans_by_page.get(page_number, []))
        pages.append(
            ExtractedPage(
                page_number=page_number,
                spans=spans,
                char_count=sum(len(span.text) for span in spans),
            )
        )
    return ExtractedDocument(
        document_id=document_id,
        filename=filename,
        pages=tuple(pages),
        extractor_version=extractor_version,
    )
