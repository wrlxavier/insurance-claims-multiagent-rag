#!/usr/bin/env python3
"""Chunk the parsed-clause corpus -- ``build/chunks.{parquet,jsonl}`` -- [M3-01].

Reads every document's clause tree from ``data/cache/clause_trees/`` (this
script never re-segments -- a missing cache means
``scripts/build_clause_tree.py`` has not been run yet, and silently
re-deriving it here would hide that instead of failing loudly, mirroring
``scripts/build_corpus.py``'s own convention), classifies each clause the
same way ``scripts/build_corpus.py`` does (cache-backed, so this re-run is
free once ``data/cache/llm_classification/cache.jsonl`` is warm), chunks
the result via [application.use_cases.chunking.chunk_typed_clauses],
flattens against [infrastructure.rag.chunk_schema.ChunkRecord], and writes
the combined chunk corpus plus a per-document Markdown report and build
manifest.

Not part of ``make parse`` -- ``build/parsed_clauses.*`` is the frozen
contract M2's golden-set/eval tooling already depends on, and forcing every
``make parse`` run to also chunk would slow that loop for no benefit until
[M3-02] (embeddings) actually consumes ``build/chunks.*``. Run via
``make build-chunks`` after ``make build-clause-tree``.
"""

from pathlib import Path
from typing import Literal

from tqdm import tqdm

from application.use_cases.chunking import CHUNKING_VERSION, chunk_typed_clauses
from application.use_cases.clause_classification import classify_and_enrich_clauses
from application.use_cases.clause_segmentation import CLAUSE_SEGMENTATION_VERSION
from domain.clause_tree import ClauseTree
from infrastructure.config.llm_client_factory import build_chat_model
from infrastructure.config.settings import get_chunking_settings, get_llm_settings
from infrastructure.parsing.clause_tree_caching import (
    clause_tree_cache_path,
    compute_clause_tree_cache_key,
    read_clause_tree_cache,
)
from infrastructure.parsing.llm_classification_cache import CachingClauseClassifier
from infrastructure.parsing.llm_classifier import LangchainClauseClassifier
from infrastructure.parsing.manifest import read_manifest
from infrastructure.parsing.rules_loader import load_classification_rules
from infrastructure.rag.chunk_artifact import (
    CHUNKS_JSONL_PATH,
    CHUNKS_MANIFEST_PATH,
    CHUNKS_PARQUET_PATH,
    ChunksBuildManifest,
    utc_now,
    write_chunks_jsonl,
    write_chunks_manifest,
    write_chunks_parquet,
)
from infrastructure.rag.chunk_report import ChunkReportRow, render_chunk_report
from infrastructure.rag.chunk_schema import SCHEMA_VERSION, ChunkRecord, flatten_chunk

MANIFEST_PATH = Path("data/policies/manifest.csv")
RULES_PATH = Path("data/parsing/clause_type_mapping.csv")
LLM_CLASSIFICATION_CACHE_PATH = Path("data/cache/llm_classification/cache.jsonl")
REPORT_PATH = Path("docs/CHUNKING_REPORT.md")


def resolve_source(extraction_mode: str) -> Literal["text", "ocr"]:
    """Map the manifest's ``extraction_mode`` to the chunk schema's ``source``.

    Mirrors ``scripts/build_corpus.py``'s ``resolve_source`` exactly: a whole
    document is routed through one extraction path, so this is decided once per
    document, never per clause.
    """
    return "ocr" if extraction_mode == "ocr_required" else "text"


def load_clause_tree(document_id: str, filename: str) -> ClauseTree:
    """Load a document's clause tree from its upstream [M1-04] cache.

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
    return read_clause_tree_cache(path)


def run_build_chunks() -> tuple[list[ChunkRecord], list[ChunkReportRow]]:
    """Chunk every document's clause tree. Returns (records, report_rows)."""
    manifest_records = read_manifest(MANIFEST_PATH)
    rules = load_classification_rules(RULES_PATH)
    chunking_settings = get_chunking_settings()
    llm_settings = get_llm_settings()
    llm = build_chat_model(
        llm_settings,
        llm_settings.llm_model_fast,
        provider_order=llm_settings.llm_classification_provider_order,
        allow_fallbacks=llm_settings.llm_classification_allow_fallbacks,
    )
    classifier = CachingClauseClassifier(
        LangchainClauseClassifier(llm),
        model=llm_settings.llm_model_fast,
        cache_path=LLM_CLASSIFICATION_CACHE_PATH,
    )

    records: list[ChunkRecord] = []
    rows: list[ChunkReportRow] = []
    for entry in tqdm(manifest_records, desc="Documents", unit="doc"):
        document_id = entry["id"]
        filename = entry["filename"]
        source = resolve_source(entry["extraction_mode"])
        tree = load_clause_tree(document_id, filename)
        typed_clauses = classify_and_enrich_clauses(
            tree,
            manifest_records,
            rules,
            classifier,
            max_workers=llm_settings.llm_classification_max_workers,
        )
        chunks, report = chunk_typed_clauses(
            typed_clauses,
            min_char_count=chunking_settings.chunk_min_char_count,
            target_char_count=chunking_settings.chunk_target_char_count,
            max_char_count=chunking_settings.chunk_max_char_count,
            sliding_window_overlap_chars=chunking_settings.chunk_sliding_window_overlap_chars,
        )
        records.extend(flatten_chunk(chunk, source=source) for chunk in chunks)
        rows.append(
            ChunkReportRow(
                document_id=document_id,
                filename=filename,
                clause_count=tree.report.clause_count,
                chunk_count=report.chunk_count,
                single_count=report.single_count,
                merged_count=report.merged_count,
                item_boundary_split_count=report.item_boundary_split_count,
                sliding_window_split_count=report.sliding_window_split_count,
                min_char_count=report.min_char_count,
                p50_char_count=report.p50_char_count,
                p90_char_count=report.p90_char_count,
                max_char_count=report.max_char_count,
            )
        )
        print(f"{filename}: chunks={report.chunk_count}")

    return records, rows


def main() -> None:
    """Chunk the corpus, writing it plus a report and manifest under ``build/``."""
    records, rows = run_build_chunks()

    write_chunks_parquet(records, CHUNKS_PARQUET_PATH)
    write_chunks_jsonl(records, CHUNKS_JSONL_PATH)
    REPORT_PATH.write_text(render_chunk_report(rows), encoding="utf-8")
    manifest = ChunksBuildManifest(
        schema_version=SCHEMA_VERSION,
        chunking_version=CHUNKING_VERSION,
        clause_segmentation_version=CLAUSE_SEGMENTATION_VERSION,
        built_at_utc=utc_now(),
        chunk_counts_by_document={row.document_id: row.chunk_count for row in rows},
        total_chunk_count=len(records),
    )
    write_chunks_manifest(manifest, CHUNKS_MANIFEST_PATH)
    print(
        f"Wrote {len(records)} chunks across {len(rows)} documents to "
        f"{CHUNKS_PARQUET_PATH}, {CHUNKS_JSONL_PATH}, {REPORT_PATH}, "
        f"{CHUNKS_MANIFEST_PATH}"
    )


if __name__ == "__main__":
    main()
