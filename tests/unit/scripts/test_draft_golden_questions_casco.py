"""Tests for the CASCO drafting/repair script's pure logic [M2-02].

Selection logic lives in `test_casco_clause_selection.py`; this file covers
the orchestration layer -- context bundles, the provider heuristic, CSV
merge behaviour and near-duplicate detection. No LLM calls.
"""

import csv
from pathlib import Path

import pytest
from scripts.draft_golden_questions_casco import (
    CSV_FIELDNAMES,
    MAX_BUNDLE_CLAUSES,
    build_context_bundle,
    cache_key,
    display_term,
    find_near_duplicates,
    format_clause_library,
    load_existing_csv,
    looks_like_quota_exhaustion,
    sort_key_for_row_id,
    write_csv,
)

from infrastructure.parsing.clause_schema import ParsedClauseRecord


def make_record(**overrides: object) -> ParsedClauseRecord:
    """Build a valid ParsedClauseRecord, overridable per test."""
    fields: dict[str, object] = {
        "schema_version": "v1",
        "clause_id": "15:10-danos-aos-vidros-basica",
        "document_id": "15",
        "parent_id": None,
        "path": "10-danos-aos-vidros-basica",
        "title": "10. DANOS AOS VIDROS",
        "text": "Garante a reparação ou substituição dos vidros do veículo.",
        "clause_type": "coverage",
        "type_source": "rule",
        "confidence": None,
        "bundle_section": "CONDIÇÕES ESPECIAIS",
        "page_start": 1,
        "page_end": 1,
        "source": "text",
        "boundary_source": None,
        "susep_process": "123",
        "insurer": "Mapfre",
        "cnpj": "00.000.000/0001-00",
        "product_line": "CASCO",
        "indemnity_regime": "VD+VMR",
        "filing_year": "2024",
    }
    fields.update(overrides)
    return ParsedClauseRecord.model_validate(fields)


def index(records: list[ParsedClauseRecord]) -> dict[str, ParsedClauseRecord]:
    """Index records by clause_id."""
    return {record.clause_id: record for record in records}


# --- context bundles: what the model is allowed to reference --------------


@pytest.mark.unit
def test_bundle_starts_with_the_primary_clause() -> None:
    primary = make_record(clause_id="15:10")
    bundle = build_context_bundle([primary], index([primary]), {}, {}, "15:10")
    assert bundle[0] == "15:10"


@pytest.mark.unit
def test_bundle_includes_children_so_a_container_can_defer_to_them() -> None:
    primary = make_record(clause_id="15:10", bundle_section=None)
    child = make_record(
        clause_id="15:10/10.2", parent_id="15:10", path="10/10.2", bundle_section=None
    )
    records = [primary, child]
    bundle = build_context_bundle(
        records, index(records), {"15:10": ["15:10/10.2"]}, {}, "15:10"
    )
    assert "15:10/10.2" in bundle


@pytest.mark.unit
def test_bundle_includes_the_parent_for_context() -> None:
    parent = make_record(clause_id="15:10", bundle_section=None)
    primary = make_record(
        clause_id="15:10/10.2", parent_id="15:10", path="10/10.2", bundle_section=None
    )
    records = [parent, primary]
    bundle = build_context_bundle(records, index(records), {}, {}, "15:10/10.2")
    assert "15:10" in bundle


@pytest.mark.unit
def test_bundle_includes_byte_identical_twins() -> None:
    primary = make_record(clause_id="9:1-2", bundle_section=None)
    twin = make_record(clause_id="9:1-4", path="1-4", bundle_section=None)
    records = [primary, twin]
    bundle = build_context_bundle(
        records, index(records), {}, {"9:1-2": frozenset({"9:1-4"})}, "9:1-2"
    )
    assert "9:1-4" in bundle


@pytest.mark.unit
def test_bundle_never_repeats_a_clause_id() -> None:
    primary = make_record(clause_id="15:10", bundle_section=None)
    sibling = make_record(clause_id="15:11", path="11", bundle_section=None)
    records = [primary, sibling]
    bundle = build_context_bundle(
        records,
        index(records),
        {"15:10": ["15:11"]},
        {"15:10": frozenset({"15:11"})},
        "15:10",
    )
    assert len(bundle) == len(set(bundle))


@pytest.mark.unit
def test_bundle_is_capped() -> None:
    primary = make_record(clause_id="15:0", path="0", parent_id="15:root")
    siblings = [
        make_record(clause_id=f"15:{i}", path=str(i), parent_id="15:root")
        for i in range(1, 30)
    ]
    records = [primary, *siblings]
    bundle = build_context_bundle(records, index(records), {}, {}, "15:0")
    assert len(bundle) <= MAX_BUNDLE_CLAUSES


@pytest.mark.unit
def test_format_clause_library_labels_every_clause_by_id() -> None:
    records = [
        make_record(clause_id="15:10"),
        make_record(clause_id="15:11", path="11"),
    ]
    rendered = format_clause_library(["15:10", "15:11"], index(records))
    assert "[15:10]" in rendered
    assert "[15:11]" in rendered


# --- provider heuristic ---------------------------------------------------


@pytest.mark.unit
def test_looks_like_quota_exhaustion_matches_credit_balance() -> None:
    assert looks_like_quota_exhaustion(RuntimeError("Your credit balance is too low"))


@pytest.mark.unit
def test_looks_like_quota_exhaustion_ignores_transient_errors() -> None:
    assert not looks_like_quota_exhaustion(TimeoutError("Connection timed out"))
    assert not looks_like_quota_exhaustion(RuntimeError("503 service unavailable"))


@pytest.mark.unit
def test_cache_key_is_stable_and_content_sensitive() -> None:
    assert cache_key("abc") == cache_key("abc")
    assert cache_key("abc") != cache_key("abd")


# --- near-duplicate detection (review code E7) ----------------------------


@pytest.mark.unit
def test_find_near_duplicates_flags_a_verbatim_repeat() -> None:
    question = "Segundo a apólice, qual é a diferença entre furto e roubo?"
    matches = find_near_duplicates({"1-00": question, "2-00": question})
    assert matches == {"2-00": "1-00"}


@pytest.mark.unit
def test_find_near_duplicates_ignores_distinct_questions() -> None:
    matches = find_near_duplicates(
        {
            "1-00": "O incêndio do veículo está coberto?",
            "2-00": "Qual é o prazo para avisar o sinistro à seguradora?",
        }
    )
    assert matches == {}


# --- CSV round-tripping ---------------------------------------------------


@pytest.mark.unit
def test_sort_key_orders_by_document_then_slot_numerically() -> None:
    row_ids = ["10-01", "2-00", "10-00", "2-01"]
    assert sorted(row_ids, key=sort_key_for_row_id) == [
        "2-00",
        "2-01",
        "10-00",
        "10-01",
    ]


@pytest.mark.unit
def test_load_existing_csv_returns_empty_dict_when_file_missing(tmp_path: Path) -> None:
    assert load_existing_csv(tmp_path / "missing.csv") == {}


@pytest.mark.unit
def test_write_csv_roundtrips_and_preserves_review_state(tmp_path: Path) -> None:
    path = tmp_path / "draft.csv"
    row = dict.fromkeys(CSV_FIELDNAMES, "")
    row["row_id"] = "1-00"
    row["question"] = "Pergunta original"
    row["approved"] = "Y"
    row["authored_at"] = "2026-08-22"
    write_csv(path, {"1-00": row})

    reloaded = load_existing_csv(path)
    assert reloaded["1-00"]["approved"] == "Y"
    assert reloaded["1-00"]["authored_at"] == "2026-08-22"


@pytest.mark.unit
def test_write_csv_orders_rows_by_document_then_slot(tmp_path: Path) -> None:
    path = tmp_path / "draft.csv"
    rows = {}
    for row_id in ("10-01", "2-00", "10-00", "2-01"):
        row = dict.fromkeys(CSV_FIELDNAMES, "")
        row["row_id"] = row_id
        rows[row_id] = row
    write_csv(path, rows)

    with path.open(newline="", encoding="utf-8") as handle:
        assert [r["row_id"] for r in csv.DictReader(handle)] == [
            "2-00",
            "2-01",
            "10-00",
            "10-01",
        ]


# --- round-3: semantic near-duplicate detection ---------------------------


@pytest.mark.unit
def test_near_duplicates_catch_a_paraphrase_the_ratio_misses() -> None:
    """3-00 vs 6-00 scored 0.648 lexically and slipped past a 0.85 threshold."""
    matches = find_near_duplicates(
        {
            "3-00": "Como a apólice define o Limite Máximo de Indenização (LMI)?",
            "6-00": (
                "Como a apólice define o Limite Máximo de Indenização (LMI) e o "
                "que esse valor representa em relação à cobertura contratada?"
            ),
        }
    )
    assert matches == {"6-00": "3-00"}


@pytest.mark.unit
def test_near_duplicates_leave_distinct_topics_alone() -> None:
    assert (
        find_near_duplicates(
            {
                "1-00": "O incêndio do veículo em garagem está coberto?",
                "2-00": "Qual é o prazo para comunicar o sinistro à seguradora?",
            }
        )
        == {}
    )


@pytest.mark.unit
def test_display_term_never_leaks_the_internal_key() -> None:
    assert display_term("roubo_furto") == "roubo ou furto"
    assert "_" not in display_term("rc_facultativa")
