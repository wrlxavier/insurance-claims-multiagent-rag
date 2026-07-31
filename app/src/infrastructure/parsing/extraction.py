"""PyMuPDF-based text extraction, isolated behind the TextExtractor port.

PyMuPDF (AGPLv3 / commercial dual-licensed) is a deliberate, accepted
dependency for this project -- see ``docs/LICENSING.md``. Nothing outside
this module (and its sibling parsing modules) imports ``fitz`` directly;
``domain`` and ``application`` only see the port and the entities in
[domain.extracted_text].
"""

from pathlib import Path
from typing import Any

import fitz

from application.ports.text_extractor import TextExtractor
from domain.extracted_text import ExtractedDocument, ExtractedPage, ExtractedSpan
from infrastructure.parsing.normalization import normalize_text, should_rejoin

PYMUPDF_VERSION = fitz.pymupdf_version

_TEXT_BLOCK_TYPE = 0

_RawSpan = dict[str, Any]
_RawLine = list[_RawSpan]


class PyMuPdfTextExtractor(TextExtractor):
    """Extracts position-aware text via PyMuPDF's ``get_text("dict")``."""

    def extract(self, pdf_path: Path, document_id: str) -> ExtractedDocument:
        """Extract normalized, position-aware text from ``pdf_path``."""
        document = fitz.open(pdf_path)
        try:
            pages = tuple(_extract_page(page, document_id) for page in document)
        finally:
            document.close()
        return ExtractedDocument(
            document_id=document_id,
            filename=pdf_path.name,
            pages=pages,
            extractor_version=PYMUPDF_VERSION,
        )


def _extract_page(page: fitz.Page, document_id: str) -> ExtractedPage:
    """Build one page's spans, merging hyphenated line-wraps first."""
    raw_lines = _flatten_lines(page)
    merged_lines = _merge_hyphenated_line_breaks(raw_lines)
    page_number = page.number + 1
    spans = tuple(_build_spans(document_id, page_number, merged_lines))
    char_count = sum(len(span.text) for span in spans)
    return ExtractedPage(page_number=page_number, spans=spans, char_count=char_count)


def _flatten_lines(page: fitz.Page) -> list[_RawLine]:
    """Return each visual line on the page as raw span dicts.

    Lines are in reading order, across block boundaries, and image blocks
    are skipped.
    """
    lines: list[_RawLine] = []
    text_dict: dict[str, Any] = page.get_text("dict")
    for block in text_dict["blocks"]:
        if block.get("type") != _TEXT_BLOCK_TYPE:
            continue
        for line in block["lines"]:
            spans = line.get("spans", [])
            if spans:
                lines.append(spans)
    return lines


def _merge_hyphenated_line_breaks(lines: list[_RawLine]) -> list[_RawLine]:
    """Merge a wrapped word's line-end hyphen with the next line's start.

    Uses ``should_rejoin`` to decide. PyMuPDF's line grouping does not
    always align with block boundaries in this corpus -- a wrapped word's
    continuation can start a new block -- so this walks the page-level line
    sequence rather than staying inside one block.
    """
    result: list[_RawLine] = []
    consume_next_first_span = False
    for index, line in enumerate(lines):
        current = list(line[1:] if consume_next_first_span else line)
        consume_next_first_span = False
        if not current:
            continue
        has_next_line = index + 1 < len(lines) and lines[index + 1]
        if has_next_line:
            current_text = "".join(normalize_text(span["text"]) for span in current)
            next_line = lines[index + 1]
            next_text = "".join(normalize_text(span["text"]) for span in next_line)
            if should_rejoin(current_text, next_text):
                last_span = dict(current[-1])
                first_next_span = next_line[0]
                joined_prefix = normalize_text(last_span["text"]).rstrip()[:-1]
                last_span["text"] = joined_prefix + normalize_text(
                    first_next_span["text"]
                )
                current[-1] = last_span
                consume_next_first_span = True
        result.append(current)
    return result


def _build_spans(
    document_id: str, page_number: int, lines: list[_RawLine]
) -> list[ExtractedSpan]:
    """Flatten merged lines into page-ordered, normalized spans."""
    spans: list[ExtractedSpan] = []
    order = 0
    for line_id, line in enumerate(lines):
        for raw_span in line:
            bbox = raw_span["bbox"]
            spans.append(
                ExtractedSpan(
                    document_id=document_id,
                    page_number=page_number,
                    line_id=line_id,
                    order=order,
                    bbox=(
                        float(bbox[0]),
                        float(bbox[1]),
                        float(bbox[2]),
                        float(bbox[3]),
                    ),
                    font_size=float(raw_span["size"]),
                    font_name=str(raw_span["font"]),
                    text=normalize_text(str(raw_span["text"])),
                )
            )
            order += 1
    return spans
