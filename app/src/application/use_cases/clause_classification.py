"""Use case for classifying clauses and attaching provenance."""

import re
import time
import unicodedata
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

# tqdm is a pragmatic UI concern in an otherwise UI-agnostic use case --
# accepted per M1-08b for build_corpus.py's long-running CLI progress needs.
from tqdm import tqdm

from application.ports.clause_classifier import ClauseClassifierPort
from domain.clause_classification import (
    ClauseProvenance,
    ClauseType,
    MissingProvenanceError,
    TypedClause,
    TypeSource,
)
from domain.clause_tree import Clause, ClauseTree

CLASSIFICATION_MAX_ATTEMPTS = 3
CLASSIFICATION_RETRY_DELAY_SECONDS = 5.0


def normalize_heading(text: str) -> str:
    """Normalize heading for rule matching (lowercase, no accents, stripped)."""
    if not text:
        return ""
    text = text.lower().strip()
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")
    return text


def build_provenance(
    document_id: str, manifest_records: list[dict[str, str]]
) -> ClauseProvenance:
    """Build provenance from manifest records using CNPJ logic."""
    for record in manifest_records:
        if record["id"] == document_id:
            return ClauseProvenance(
                document_id=document_id,
                susep_process=record.get("susep_process", ""),
                insurer=record.get("insurer", ""),
                cnpj=record.get("cnpj", ""),
                product_line=record.get("product_line", ""),
                indemnity_regime=record.get("indemnity_regime", ""),
                process_year=record.get("process_year", ""),
            )
    raise MissingProvenanceError(document_id)


def _classify_with_fallback(
    clause: Clause,
    classifier: ClauseClassifierPort,
    *,
    max_attempts: int = CLASSIFICATION_MAX_ATTEMPTS,
    retry_delay_seconds: float = CLASSIFICATION_RETRY_DELAY_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[ClauseType, float]:
    """Call the LLM port, retrying transient failures before falling back.

    Retries up to ``max_attempts`` times, sleeping ``retry_delay_seconds``
    between attempts, before falling back to OTHER/0.0 -- no exception ever
    propagates out of this function.
    """
    full_text = "\n".join(clause.content_lines)
    for attempt in range(1, max_attempts + 1):
        try:
            return classifier.classify(clause.title, full_text)
        except Exception:
            if attempt < max_attempts:
                sleep(retry_delay_seconds)
    return ClauseType.OTHER, 0.0


def classify_and_enrich_clauses(
    tree: ClauseTree,
    manifest_records: list[dict[str, str]],
    rules: list[tuple[re.Pattern[str], ClauseType]],
    classifier: ClauseClassifierPort,
    max_workers: int = 1,
    *,
    sleep: Callable[[float], None] = time.sleep,
) -> list[TypedClause]:
    """Process the clause tree to classify each clause and attach provenance.

    Args:
        tree: The ClauseTree segment output from M1-04.
        manifest_records: The parsed manifest.csv rows.
        rules: Ordered list of (compiled_regex, ClauseType) for deterministic pass.
        classifier: The LLM port for fallback classification.
        max_workers: Thread-pool size for the LLM fallback pass. The rule
            pass is always sequential (no I/O). ``1`` (the default) runs the
            LLM pass sequentially too -- callers that pass a classifier
            backed by a real network call should raise this to shorten the
            LLM pass's wall-clock time.
        sleep: Sleep function used between retry attempts in the LLM pass --
            overridable in tests to avoid real delays.

    Returns:
        List of TypedClause.
    """
    provenance = build_provenance(tree.document_id, manifest_records)

    # First pass: deterministic rules -- always sequential, no I/O.
    rule_matches: dict[str, ClauseType] = {}
    pending: list[Clause] = []
    for clause in tree.all_clauses:
        normalized_title = normalize_heading(clause.title)
        for pattern, mapped_type in rules:
            if pattern.search(normalized_title):
                rule_matches[clause.clause_id] = mapped_type
                break
        else:
            pending.append(clause)

    # Second pass: LLM fallback for whatever the rules left unmatched.
    llm_results: dict[str, tuple[ClauseType, float]] = {}
    if pending:
        if max_workers > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                outcomes = list(
                    tqdm(
                        pool.map(
                            lambda clause: _classify_with_fallback(
                                clause, classifier, sleep=sleep
                            ),
                            pending,
                        ),
                        total=len(pending),
                        desc="Clauses (LLM)",
                        unit="clause",
                        leave=False,
                    )
                )
        else:
            outcomes = [
                _classify_with_fallback(clause, classifier, sleep=sleep)
                for clause in tqdm(
                    pending, desc="Clauses (LLM)", unit="clause", leave=False
                )
            ]
        for clause, outcome in zip(pending, outcomes, strict=True):
            llm_results[clause.clause_id] = outcome

    typed_clauses = []
    for clause in tree.all_clauses:
        if clause.clause_id in rule_matches:
            clause_type = rule_matches[clause.clause_id]
            type_source = TypeSource.RULE
            confidence = 1.0
        else:
            clause_type, confidence = llm_results[clause.clause_id]
            type_source = TypeSource.LLM

        typed_clauses.append(
            TypedClause(
                clause=clause,
                clause_type=clause_type,
                type_source=type_source,
                confidence=confidence,
                provenance=provenance,
            )
        )

    return typed_clauses
