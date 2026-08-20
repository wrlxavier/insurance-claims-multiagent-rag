#!/usr/bin/env python3
"""Build the final, versioned parsed-clause corpus -- ``build/`` (gitignored).

The last stage of the parsing pipeline [M1-07]: reads every document's
clause tree from ``data/cache/clause_trees/`` (this script never
re-segments -- a missing cache means ``scripts/build_clause_tree.py`` has
not been run yet, and silently re-deriving it here would hide that instead
of failing loudly), classifies each clause ([M1-05]'s deterministic rules,
falling back to a real LLM pass -- see [infrastructure.parsing.
llm_classifier.LangchainClauseClassifier] -- for whatever the rules leave
as ``other``), flattens the result against [infrastructure.parsing.
clause_schema.ParsedClauseRecord], and writes the combined corpus plus a
build manifest recording exactly what produced it.

The LLM pass is the slow, non-free part: it runs a configurable number of
clauses concurrently per document (``LLM_CLASSIFICATION_MAX_WORKERS`` in
``.env``, default 10), and every result is cached to disk by [infrastructure.parsing.
llm_classification_cache.CachingClauseClassifier] keyed by (model, title,
text) -- a kill/crash mid-run loses at most the in-flight batch, since a
rerun skips whatever is already cached.

Run via ``make parse``, which chains this after ``extract-text``,
``remove-boilerplate`` and ``build-clause-tree`` -- the whole pipeline,
from ``data/policies/raw/`` to ``build/``, in one command.
"""

from pathlib import Path
from typing import Literal

from tqdm import tqdm

from application.use_cases.boilerplate_removal import BOILERPLATE_REMOVAL_VERSION
from application.use_cases.clause_classification import classify_and_enrich_clauses
from application.use_cases.clause_segmentation import CLAUSE_SEGMENTATION_VERSION
from domain.clause_tree import ClauseTree
from infrastructure.config.llm_client_factory import build_chat_model
from infrastructure.config.settings import get_llm_settings
from infrastructure.parsing.clause_schema import (
    SCHEMA_VERSION,
    ParsedClauseRecord,
    flatten_typed_clause,
)
from infrastructure.parsing.clause_tree_caching import (
    clause_tree_cache_path,
    compute_clause_tree_cache_key,
    read_clause_tree_cache,
)
from infrastructure.parsing.corpus_artifact import (
    BUILD_MANIFEST_PATH,
    JSONL_PATH,
    PARQUET_PATH,
    BuildManifest,
    utc_now,
    write_build_manifest,
    write_parsed_clauses_jsonl,
    write_parsed_clauses_parquet,
)
from infrastructure.parsing.llm_classification_cache import CachingClauseClassifier
from infrastructure.parsing.llm_classifier import LangchainClauseClassifier
from infrastructure.parsing.manifest import read_manifest
from infrastructure.parsing.rules_loader import load_classification_rules

MANIFEST_PATH = Path("data/policies/manifest.csv")
RULES_PATH = Path("data/parsing/clause_type_mapping.csv")
LLM_CLASSIFICATION_CACHE_PATH = Path("data/cache/llm_classification/cache.jsonl")


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


def resolve_source(extraction_mode: str) -> Literal["text", "ocr"]:
    """Map the manifest's ``extraction_mode`` to the flat schema's ``source``.

    A whole document is routed through exactly one extraction path (see
    ``scripts/extract_text.py``), so this is decided once per document,
    never per clause.
    """
    return "ocr" if extraction_mode == "ocr_required" else "text"


def run_build_corpus() -> tuple[list[ParsedClauseRecord], dict[str, int]]:
    """Classify and flatten every document's clause tree. Returns (records, counts)."""
    manifest_records = read_manifest(MANIFEST_PATH)
    rules = load_classification_rules(RULES_PATH)
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

    records: list[ParsedClauseRecord] = []
    counts: dict[str, int] = {}
    for entry in tqdm(manifest_records, desc="Documents", unit="doc"):
        document_id = entry["id"]
        filename = entry["filename"]
        tree = load_clause_tree(document_id, filename)
        typed_clauses = classify_and_enrich_clauses(
            tree,
            manifest_records,
            rules,
            classifier,
            max_workers=llm_settings.llm_classification_max_workers,
        )
        source = resolve_source(entry["extraction_mode"])
        document_records = [
            flatten_typed_clause(typed, source=source) for typed in typed_clauses
        ]
        records.extend(document_records)
        counts[document_id] = len(document_records)
        print(f"{filename}: clauses={len(document_records)}")

    return records, counts


def main() -> None:
    """Build the corpus and write it, plus a build manifest, under ``build/``."""
    records, counts = run_build_corpus()

    write_parsed_clauses_parquet(records, PARQUET_PATH)
    write_parsed_clauses_jsonl(records, JSONL_PATH)
    manifest = BuildManifest(
        schema_version=SCHEMA_VERSION,
        clause_segmentation_version=CLAUSE_SEGMENTATION_VERSION,
        boilerplate_removal_version=BOILERPLATE_REMOVAL_VERSION,
        llm_classification_enabled=True,
        built_at_utc=utc_now(),
        clause_counts_by_document=counts,
        total_clause_count=len(records),
    )
    write_build_manifest(manifest, BUILD_MANIFEST_PATH)
    print(
        f"Wrote {len(records)} clauses across {len(counts)} documents to "
        f"{PARQUET_PATH}, {JSONL_PATH}, {BUILD_MANIFEST_PATH}"
    )


if __name__ == "__main__":
    main()
