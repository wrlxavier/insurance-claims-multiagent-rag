"""Tests for the adversarial golden-set drafting script [M2-03].

No live LLM calls -- these cover the pure functions: row_id ordering, slot
grouping, context-bundle assembly (the distractor must reach the prompt's
clause library but never the allowed-ids list), CSV row population, and the
DoD coverage tally. Matches this repo's convention of mocking the chat model
rather than calling OpenRouter in the test suite.
"""

from collections import defaultdict
from typing import cast

import pytest
from scripts.adversarial_clause_selection import AdversarialSlot
from scripts.draft_golden_questions_adversarial import (
    CATEGORY_QUESTION_TYPE,
    DraftableQuestionType,
    _rows_from_slots,
    build_draft_prompt,
    build_slot_context,
    group_slots,
    print_coverage_report,
    sort_key_for_row_id,
)

from infrastructure.parsing.clause_schema import ParsedClauseRecord


def make_record(**overrides: object) -> ParsedClauseRecord:
    """Build a valid ParsedClauseRecord, overridable per test."""
    fields: dict[str, object] = {
        "schema_version": "v1",
        "clause_id": "1:riscos-cobertos",
        "document_id": "1",
        "parent_id": None,
        "path": "riscos-cobertos",
        "title": "RISCOS COBERTOS",
        "text": "Garante os prejuízos decorrentes de colisão do veículo segurado.",
        "clause_type": "coverage",
        "type_source": "rule",
        "confidence": None,
        "bundle_section": None,
        "page_start": 1,
        "page_end": 1,
        "source": "text",
        "boundary_source": None,
        "susep_process": "123",
        "insurer": "Test Seguros",
        "cnpj": "00000000000100",
        "product_line": "CASCO",
        "indemnity_regime": "VD",
        "filing_year": "2024",
    }
    fields.update(overrides)
    return ParsedClauseRecord.model_validate(fields)


# --- sort_key_for_row_id -----------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    ("row_id", "expected"),
    [
        ("cwe-1-00", ("cwe", "1", 0)),
        ("xdoc-4x18-01", ("xdoc", "4x18", 1)),
        ("hdi-12x21-00", ("hdi", "12x21", 0)),
        ("bundle-10-03", ("bundle", "10", 3)),
    ],
)
def test_sort_key_for_row_id_parses_category_doc_and_slot(
    row_id: str, expected: tuple[str, str, int]
) -> None:
    assert sort_key_for_row_id(row_id) == expected


@pytest.mark.unit
def test_sort_key_for_row_id_orders_by_category_then_slot() -> None:
    ids = ["xdoc-4x18-01", "cwe-1-00", "xdoc-4x18-00", "bundle-10-00"]
    assert sorted(ids, key=sort_key_for_row_id) == [
        "bundle-10-00",
        "cwe-1-00",
        "xdoc-4x18-00",
        "xdoc-4x18-01",
    ]


# --- group_slots --------------------------------------------------------


@pytest.mark.unit
def test_group_slots_groups_coverage_with_exclusion_by_document() -> None:
    slots = [
        AdversarialSlot(
            row_id="cwe-1-00",
            adversarial_category="coverage_with_exclusion",
            document_id="1",
            primary_clause_id="1:a",
            secondary_clause_id="1:b",
        ),
        AdversarialSlot(
            row_id="cwe-1-01",
            adversarial_category="coverage_with_exclusion",
            document_id="1",
            primary_clause_id="1:c",
            secondary_clause_id="1:d",
        ),
    ]
    groups = group_slots(slots)
    assert groups == {"cwe-1": slots}


@pytest.mark.unit
def test_group_slots_groups_cross_document_by_pair_prefix() -> None:
    slots = [
        AdversarialSlot(
            row_id="xdoc-4x18-00",
            adversarial_category="cross_document",
            document_id="4",
            primary_clause_id="4:a",
            distractor_clause_id="18:a",
        ),
        AdversarialSlot(
            row_id="xdoc-4x18-01",
            adversarial_category="cross_document",
            document_id="4",
            primary_clause_id="4:b",
            distractor_clause_id="18:b",
        ),
    ]
    groups = group_slots(slots)
    assert set(groups) == {"xdoc-4x18"}
    assert len(groups["xdoc-4x18"]) == 2


# --- build_slot_context ---------------------------------------------------


@pytest.mark.unit
def test_build_slot_context_shows_distractor_in_library_but_not_allowed_ids() -> None:
    primary = make_record(clause_id="4:a", document_id="4")
    distractor = make_record(clause_id="18:a", document_id="18")
    records = [primary, distractor]
    by_id = {r.clause_id: r for r in records}
    slot = AdversarialSlot(
        row_id="xdoc-4x18-00",
        adversarial_category="cross_document",
        document_id="4",
        primary_clause_id="4:a",
        distractor_clause_id="18:a",
    )
    ctx = build_slot_context(
        slot,
        records=records,
        by_id=by_id,
        children_by_parent=defaultdict(list),
        twins={},
    )
    allowed_ids = cast(list[str], ctx["allowed_ids"])
    library_ids = cast(list[str], ctx["library_ids"])
    assert "18:a" not in allowed_ids
    assert "18:a" in library_ids
    assert "4:a" in allowed_ids


@pytest.mark.unit
def test_build_slot_context_adds_hdi_entity_metadata() -> None:
    primary = make_record(
        clause_id="12:a", document_id="12", insurer="HDI Seguros", cnpj="29980158000157"
    )
    distractor = make_record(
        clause_id="21:a", document_id="21", insurer="HDI Global", cnpj="18096627000153"
    )
    records = [primary, distractor]
    by_id = {r.clause_id: r for r in records}
    slot = AdversarialSlot(
        row_id="hdi-12x21-00",
        adversarial_category="hdi_brand_collision",
        document_id="12",
        primary_clause_id="12:a",
        distractor_clause_id="21:a",
    )
    ctx = build_slot_context(
        slot,
        records=records,
        by_id=by_id,
        children_by_parent=defaultdict(list),
        twins={},
    )
    assert ctx["cnpj_a"] == "29980158000157"
    assert ctx["cnpj_b"] == "18096627000153"


# --- build_draft_prompt ---------------------------------------------------


@pytest.mark.unit
def test_build_draft_prompt_labels_distractor_for_cross_document() -> None:
    primary = make_record(clause_id="4:a", document_id="4")
    by_id = {"4:a": primary}
    prompt = build_draft_prompt(
        category="cross_document",
        group_label="xdoc-4x18",
        library_ids=["4:a"],
        by_id=by_id,
        slots=[
            {
                "row_id": "xdoc-4x18-00",
                "allowed_ids": ["4:a"],
                "primary_clause_id": "4:a",
                "secondary_clause_id": None,
                "distractor_clause_id": "18:a",
            }
        ],
    )
    assert "CONCORRENTE" in prompt
    assert "18:a" in prompt
    assert "NUNCA em reference_clause_ids" in prompt
    assert "TIPO OBRIGATÓRIO: 'cross_document'" in prompt


@pytest.mark.unit
def test_build_draft_prompt_labels_direct_lookup_for_hdi_and_bundle() -> None:
    primary = make_record(clause_id="12:a", document_id="12")
    by_id = {"12:a": primary}
    prompt = build_draft_prompt(
        category="hdi_brand_collision",
        group_label="hdi-12x21",
        library_ids=["12:a"],
        by_id=by_id,
        slots=[
            {
                "row_id": "hdi-12x21-00",
                "allowed_ids": ["12:a"],
                "primary_clause_id": "12:a",
                "secondary_clause_id": None,
                "distractor_clause_id": "21:a",
                "insurer_a": "HDI Seguros",
                "cnpj_a": "29980158000157",
                "insurer_b": "HDI Global",
                "cnpj_b": "18096627000153",
            }
        ],
    )
    assert "TIPO OBRIGATÓRIO: 'direct_lookup'" in prompt


@pytest.mark.unit
def test_category_question_type_matches_dod_schema_mapping() -> None:
    assert CATEGORY_QUESTION_TYPE == {
        "coverage_with_exclusion": DraftableQuestionType.COVERAGE_WITH_EXCLUSION,
        "cross_document": DraftableQuestionType.CROSS_DOCUMENT,
        "hdi_brand_collision": DraftableQuestionType.DIRECT_LOOKUP,
        "bundle_section": DraftableQuestionType.DIRECT_LOOKUP,
    }


@pytest.mark.unit
def test_build_draft_prompt_requires_coverage_and_exclusion_ids() -> None:
    primary = make_record(clause_id="1:a", document_id="1")
    by_id = {"1:a": primary}
    prompt = build_draft_prompt(
        category="coverage_with_exclusion",
        group_label="cwe-1",
        library_ids=["1:a"],
        by_id=by_id,
        slots=[
            {
                "row_id": "cwe-1-00",
                "allowed_ids": ["1:a", "1:b"],
                "primary_clause_id": "1:a",
                "secondary_clause_id": "1:b",
                "distractor_clause_id": None,
            }
        ],
    )
    assert "coverage_with_exclusion" in prompt
    assert "1:b" in prompt


# --- _rows_from_slots -----------------------------------------------------


@pytest.mark.unit
def test_rows_from_slots_populates_distractor_columns() -> None:
    primary = make_record(clause_id="4:a", document_id="4", title="Título A")
    distractor = make_record(clause_id="18:a", document_id="18", title="Título B")
    by_id = {"4:a": primary, "18:a": distractor}
    slot = AdversarialSlot(
        row_id="xdoc-4x18-00",
        adversarial_category="cross_document",
        document_id="4",
        primary_clause_id="4:a",
        distractor_clause_id="18:a",
    )
    rows = _rows_from_slots([slot], by_id)
    row = rows["xdoc-4x18-00"]
    assert row["distractor_clause_id"] == "18:a"
    assert row["distractor_document_id"] == "18"
    assert row["distractor_clause_title"] == "Título B"
    assert row["adversarial_category"] == "cross_document"


@pytest.mark.unit
def test_rows_from_slots_leaves_distractor_columns_empty_for_cwe() -> None:
    primary = make_record(clause_id="1:a", document_id="1")
    by_id = {"1:a": primary}
    slot = AdversarialSlot(
        row_id="cwe-1-00",
        adversarial_category="coverage_with_exclusion",
        document_id="1",
        primary_clause_id="1:a",
        secondary_clause_id="1:b",
    )
    rows = _rows_from_slots([slot], by_id)
    row = rows["cwe-1-00"]
    assert row["distractor_clause_id"] == ""
    assert row["secondary_clause_id"] == "1:b"


# --- print_coverage_report -------------------------------------------------


@pytest.mark.unit
def test_print_coverage_report_fails_below_dod_floor(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rows = [{"adversarial_category": "bundle_section"}] * 2
    passed = print_coverage_report(rows)
    assert passed is False
    assert "BELOW DoD FLOOR" in capsys.readouterr().out


@pytest.mark.unit
def test_print_coverage_report_passes_at_dod_floor() -> None:
    rows = (
        [{"adversarial_category": "coverage_with_exclusion"}] * 15
        + [{"adversarial_category": "cross_document"}] * 10
        + [{"adversarial_category": "hdi_brand_collision"}] * 5
        + [{"adversarial_category": "bundle_section"}] * 3
    )
    assert print_coverage_report(rows) is True
