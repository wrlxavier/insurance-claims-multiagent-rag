"""Use case for classifying clauses and attaching provenance."""

import re
import unicodedata

from application.ports.clause_classifier import ClauseClassifierPort
from domain.clause_classification import (
    ClauseProvenance,
    ClauseType,
    MissingProvenanceError,
    TypedClause,
    TypeSource,
)
from domain.clause_tree import ClauseTree


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


def classify_and_enrich_clauses(
    tree: ClauseTree,
    manifest_records: list[dict[str, str]],
    rules: list[tuple[re.Pattern[str], ClauseType]],
    classifier: ClauseClassifierPort,
) -> list[TypedClause]:
    """Process the clause tree to classify each clause and attach provenance.

    Args:
        tree: The ClauseTree segment output from M1-04.
        manifest_records: The parsed manifest.csv rows.
        rules: Ordered list of (compiled_regex, ClauseType) for deterministic pass.
        classifier: The LLM port for fallback classification.

    Returns:
        List of TypedClause.
    """
    provenance = build_provenance(tree.document_id, manifest_records)
    typed_clauses = []

    for clause in tree.all_clauses:
        clause_type = None
        type_source = None
        confidence = None

        normalized_title = normalize_heading(clause.title)

        # First pass: Deterministic rules
        for pattern, mapped_type in rules:
            if pattern.search(normalized_title):
                clause_type = mapped_type
                type_source = TypeSource.RULE
                confidence = 1.0
                break

        # Second pass: LLM Fallback
        if clause_type is None:
            full_text = "\\n".join(clause.content_lines)
            try:
                clause_type, confidence = classifier.classify(clause.title, full_text)
                type_source = TypeSource.LLM
            except Exception:
                # Fallback of the fallback
                clause_type = ClauseType.OTHER
                type_source = TypeSource.LLM
                confidence = 0.0

        assert clause_type is not None
        assert type_source is not None

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
