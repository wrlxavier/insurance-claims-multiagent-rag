"""Tests for the flattened chunk schema."""

from dataclasses import replace

import pytest

from domain.chunk import Chunk, ChunkRule
from domain.clause_classification import ClauseProvenance, ClauseType, TypeSource
from infrastructure.rag.chunk_schema import SCHEMA_VERSION, flatten_chunk


def _provenance() -> ClauseProvenance:
    return ClauseProvenance(
        document_id="1",
        susep_process="15414900666201489",
        insurer="Bradesco Seguros",
        cnpj="12345678000199",
        product_line="CASCO",
        indemnity_regime="VD",
        process_year="2019",
    )


def _chunk() -> Chunk:
    return Chunk(
        document_id="1",
        chunk_id="1:2",
        clause_id="1:2",
        source_clause_ids=("1:2",),
        chunk_index=0,
        chunk_count=1,
        parent_path="1. CONDIÇÕES GERAIS",
        text="1. CONDIÇÕES GERAIS\n2. COBERTURAS\n\nTexto da cobertura.",
        char_count=56,
        rule=ChunkRule.SINGLE,
        clause_type=ClauseType.COVERAGE,
        type_source=TypeSource.RULE,
        confidence=1.0,
        bundle_section=None,
        provenance=_provenance(),
    )


@pytest.mark.unit
def test_flatten_chunk_produces_valid_record() -> None:
    chunk = _chunk()

    record = flatten_chunk(chunk, source="text")

    assert record.schema_version == SCHEMA_VERSION
    assert record.chunk_id == "1:2"
    assert record.clause_id == "1:2"
    assert record.source_clause_ids == ["1:2"]
    assert record.text == chunk.text
    assert record.source == "text"
    assert record.filing_year == "2019"
    assert record.rule == ChunkRule.SINGLE


@pytest.mark.unit
def test_flatten_chunk_preserves_multi_source_ids_and_split_rule() -> None:
    chunk = replace(
        _chunk(),
        chunk_id="1:2#0",
        source_clause_ids=("1:2", "1:2.1"),
        chunk_count=2,
        rule=ChunkRule.ITEM_BOUNDARY_SPLIT,
    )

    record = flatten_chunk(chunk, source="text")

    assert record.source_clause_ids == ["1:2", "1:2.1"]
    assert record.chunk_count == 2
    assert record.rule == ChunkRule.ITEM_BOUNDARY_SPLIT


@pytest.mark.unit
def test_flatten_chunk_carries_ocr_source() -> None:
    record = flatten_chunk(_chunk(), source="ocr")

    assert record.source == "ocr"


@pytest.mark.unit
def test_display_text_strips_only_the_ancestor_breadcrumb() -> None:
    # `_render_piece` composes `text` as `f"{parent_path}\n{anchor_title}\n\n{body}"`
    # when there are ancestors, so `display_text` is the clause keeping its own
    # heading line, with only the injected breadcrumb removed.
    record = flatten_chunk(_chunk(), source="text")

    assert record.display_text == "2. COBERTURAS\n\nTexto da cobertura."
    assert record.text == f"{record.parent_path}\n{record.display_text}"


@pytest.mark.unit
def test_display_text_equals_text_for_a_root_clause() -> None:
    chunk = replace(
        _chunk(),
        parent_path="",
        text="1. DISPOSIÇÕES GERAIS\n\nTexto.",
    )

    record = flatten_chunk(chunk, source="text")

    assert record.display_text == record.text
