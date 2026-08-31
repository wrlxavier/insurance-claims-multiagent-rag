"""Real-corpus snapshot test for [M3-01], required by its DoD.

Extracts fresh from the real PDFs in ``data/policies/raw/`` rather than
reading the Parquet caches (both gitignored and empty on a fresh clone),
mirroring ``test_clause_segmentation_corpus.py``'s pattern. Classification
uses the real deterministic rules (``data/parsing/clause_type_mapping.csv``)
plus a deterministic stub for whatever they leave unmatched -- never the
real LLM classifier -- so this stays network-free and reproducible without
depending on the gitignored ``data/cache/llm_classification/cache.jsonl``.

No snapshot-testing library exists in this repo (no syrupy, no
``__snapshots__/``); this follows the repo's own established alternative
instead: a small, human-readable JSONL fixture per document, committed
under ``snapshots/``, that a fresh run must match exactly -- so a chunking
change is visible in `git diff`, satisfying the DoD's requirement. Only a
truncated text preview is stored (not full chunk text) to keep the fixture
reviewable.

Documents 10 (Bradesco, 207pp bundle), 15 (Mapfre, 2004 legacy layout, one
huge glossary clause) and 28 (Assurant, multi-column, no bundle_section)
are the same trio already exercised together in
``test_clause_segmentation_corpus.py``, and between them exercise every
chunking rule: SINGLE and MERGED (all three have many sub-80-char
clauses), ITEM_BOUNDARY_SPLIT (docs 10/15 both have clauses over 3000
chars with lettered/bulleted items), and SLIDING_WINDOW_SPLIT (most of doc
15's ``15:glossario`` clause is term/definition prose with no
lettered/bulleted markers, so [application.use_cases.clause_segmentation.
is_list_item_line] never fires across most of it -- except its one
genuine lettered sub-list, which correctly still gets
ITEM_BOUNDARY_SPLIT).
"""

import json
from functools import cache
from pathlib import Path

import pytest

from application.use_cases.boilerplate_removal import remove_boilerplate
from application.use_cases.chunking import chunk_typed_clauses
from application.use_cases.clause_classification import classify_and_enrich_clauses
from application.use_cases.clause_segmentation import segment_document
from domain.chunk import Chunk
from domain.clause_classification import ClauseType
from infrastructure.parsing.extraction import PyMuPdfTextExtractor
from infrastructure.parsing.manifest import read_manifest
from infrastructure.parsing.rules_loader import load_classification_rules
from infrastructure.rag.chunk_schema import flatten_chunk

REPO_ROOT = Path(__file__).resolve().parents[4]
RAW_DIR = REPO_ROOT / "data" / "policies" / "raw"
MANIFEST_PATH = REPO_ROOT / "data" / "policies" / "manifest.csv"
RULES_PATH = REPO_ROOT / "data" / "parsing" / "clause_type_mapping.csv"
SNAPSHOT_DIR = Path(__file__).resolve().parent / "snapshots"

# min/target/max/overlap match ChunkingSettings' defaults
# (infrastructure.config.settings), pinned as literals here so the fixture
# stays independent of .env state, mirroring how
# test_clause_segmentation_corpus.py never touches get_parsing_settings().
_MIN_CHAR_COUNT = 150
_TARGET_CHAR_COUNT = 1800
_MAX_CHAR_COUNT = 3000
_SLIDING_WINDOW_OVERLAP_CHARS = 300

_DOCUMENTS = {
    "10": "15414900666201489.pdf",
    "15": "15414100326200483.pdf",
    "28": "15414607840202570.pdf",
}


class _RuleOnlyClassifier:
    """Deterministic stand-in for the LLM fallback pass.

    Real fallback classification hits a network LLM and depends on the
    gitignored ``data/cache/llm_classification/cache.jsonl`` to stay fast --
    neither is available on a fresh clone/CI. Every clause the deterministic
    rules pass fails to match gets OTHER/0.0 here instead, exactly like
    [application.use_cases.clause_classification._classify_with_fallback]'s
    own last-resort default when the real classifier is exhausted.
    """

    def classify(self, clause_title: str, clause_text: str) -> tuple[ClauseType, float]:
        return ClauseType.OTHER, 0.0


@cache
def _chunk_document(document_id: str, filename: str) -> list[Chunk]:
    document = PyMuPdfTextExtractor().extract(RAW_DIR / filename, document_id)
    cleaned, _counts = remove_boilerplate(document)
    tree = segment_document(cleaned)

    manifest_records = read_manifest(MANIFEST_PATH)
    rules = load_classification_rules(RULES_PATH)
    typed_clauses = classify_and_enrich_clauses(
        tree, manifest_records, rules, _RuleOnlyClassifier()
    )

    chunks, _report = chunk_typed_clauses(
        typed_clauses,
        min_char_count=_MIN_CHAR_COUNT,
        target_char_count=_TARGET_CHAR_COUNT,
        max_char_count=_MAX_CHAR_COUNT,
        sliding_window_overlap_chars=_SLIDING_WINDOW_OVERLAP_CHARS,
    )
    return chunks


def _serialize(chunk: Chunk) -> dict[str, object]:
    return {
        "chunk_id": chunk.chunk_id,
        "clause_id": chunk.clause_id,
        "source_clause_ids": list(chunk.source_clause_ids),
        "chunk_index": chunk.chunk_index,
        "chunk_count": chunk.chunk_count,
        "rule": chunk.rule.value,
        "char_count": chunk.char_count,
        "parent_path": chunk.parent_path,
        "clause_type": chunk.clause_type.value,
        "bundle_section": chunk.bundle_section,
        "text_preview": chunk.text[:120],
    }


def _load_snapshot(document_id: str) -> list[dict[str, object]]:
    path = SNAPSHOT_DIR / f"chunks_{document_id}.jsonl"
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


@pytest.mark.unit
@pytest.mark.parametrize("document_id", sorted(_DOCUMENTS))
def test_chunking_matches_committed_snapshot(document_id: str) -> None:
    chunks = _chunk_document(document_id, _DOCUMENTS[document_id])
    actual = [_serialize(chunk) for chunk in chunks]

    expected = _load_snapshot(document_id)

    assert actual == expected


@pytest.mark.unit
@pytest.mark.parametrize("document_id", sorted(_DOCUMENTS))
def test_flatten_chunk_display_text_is_the_embedded_text_minus_breadcrumb(
    document_id: str,
) -> None:
    # [M3-02]: `display_text` (citation excerpt) must be exactly the embedded
    # text with only the injected ancestor breadcrumb removed. Checked against
    # every real chunk of the trio so a change to `_render_piece` that breaks
    # the derivation is caught.
    for chunk in _chunk_document(document_id, _DOCUMENTS[document_id]):
        record = flatten_chunk(chunk, source="text")
        if chunk.parent_path:
            assert record.text == f"{chunk.parent_path}\n{record.display_text}"
        else:
            assert record.display_text == record.text


@pytest.mark.unit
def test_doc_15_glossary_clause_uses_sliding_window_split() -> None:
    # Confirms the design assumption this trio was chosen on: most of the
    # glossary's term/definition pairs carry no lettered/bulleted markers,
    # so they fall through item-boundary detection to the last-resort rule
    # -- except its one genuine lettered sub-list (the "Terceiro" entry's
    # a)-e) items), which correctly gets ITEM_BOUNDARY_SPLIT instead.
    chunks = _chunk_document("15", _DOCUMENTS["15"])

    glossary_chunks = [c for c in chunks if c.clause_id == "15:glossario"]

    assert glossary_chunks
    assert any(c.rule.value == "sliding_window_split" for c in glossary_chunks)
