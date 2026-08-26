"""Tests for clause-aware chunking."""

import pytest

from application.use_cases.chunking import chunk_typed_clauses
from domain.chunk import ChunkRule
from domain.clause_classification import (
    ClauseProvenance,
    ClauseType,
    TypedClause,
    TypeSource,
)
from domain.clause_tree import Clause, HeadingConvention


def _clause(
    clause_id: str,
    *,
    parent_id: str | None,
    title: str,
    content_lines: tuple[str, ...] = (),
    child_ids: tuple[str, ...] = (),
    bundle_section: str | None = None,
) -> Clause:
    return Clause(
        document_id="1",
        clause_id=clause_id,
        path=clause_id,
        numbering_label=clause_id,
        title=title,
        convention=HeadingConvention.NUMBERED_DECIMAL,
        depth=clause_id.count("."),
        parent_id=parent_id,
        child_ids=child_ids,
        content_lines=content_lines,
        page_start=1,
        page_end=1,
        bundle_section=bundle_section,
    )


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


def _typed(clause: Clause) -> TypedClause:
    return TypedClause(
        clause=clause,
        clause_type=ClauseType.COVERAGE,
        type_source=TypeSource.RULE,
        confidence=1.0,
        provenance=_provenance(),
    )


@pytest.mark.unit
def test_single_leaf_produces_one_chunk() -> None:
    clause = _clause(
        "1:1",
        parent_id=None,
        title="1. OBJETO",
        content_lines=("Texto do objeto do seguro." * 3,),
    )

    chunks, report = chunk_typed_clauses(
        [_typed(clause)],
        min_char_count=10,
        target_char_count=1000,
        max_char_count=1000,
        sliding_window_overlap_chars=50,
    )

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.chunk_id == "1:1"
    assert chunk.clause_id == "1:1"
    assert chunk.source_clause_ids == ("1:1",)
    assert chunk.chunk_index == 0
    assert chunk.chunk_count == 1
    assert chunk.rule == ChunkRule.SINGLE
    assert chunk.parent_path == ""
    assert chunk.text.startswith("1. OBJETO")
    assert report.chunk_count == 1
    assert report.single_count == 1


@pytest.mark.unit
def test_breadcrumb_includes_ancestor_titles_only() -> None:
    root = _clause(
        "1:9", parent_id=None, child_ids=("1:9.2",), title="9. COBERTURAS BÁSICAS"
    )
    mid = _clause(
        "1:9.2", parent_id="1:9", child_ids=("1:9.2.2",), title="9.2 R.C.F.V."
    )
    leaf = _clause(
        "1:9.2.2",
        parent_id="1:9.2",
        title="9.2.2 Riscos Cobertos",
        content_lines=(
            "Esta garantia cobrirá somente o valor que exceder o limite." * 2,
        ),
    )

    chunks, _ = chunk_typed_clauses(
        [_typed(root), _typed(mid), _typed(leaf)],
        min_char_count=1,
        target_char_count=1000,
        max_char_count=1000,
        sliding_window_overlap_chars=50,
    )

    leaf_chunk = next(c for c in chunks if c.clause_id == "1:9.2.2")
    assert leaf_chunk.parent_path == "9. COBERTURAS BÁSICAS > 9.2 R.C.F.V."
    assert leaf_chunk.text.startswith(
        "9. COBERTURAS BÁSICAS > 9.2 R.C.F.V.\n9.2.2 Riscos Cobertos"
    )


@pytest.mark.unit
def test_short_leaf_merges_into_parent() -> None:
    child = _clause(
        "1:2.1", parent_id="1:2", title="2.1 Furto", content_lines=("Curto.",)
    )
    parent = _clause(
        "1:2",
        parent_id=None,
        child_ids=("1:2.1",),
        title="2. COBERTURAS",
        content_lines=(
            "Cobertura contra danos ao veículo segurado em caso de sinistro." * 3,
        ),
    )

    chunks, _ = chunk_typed_clauses(
        [_typed(parent), _typed(child)],
        min_char_count=50,
        target_char_count=1000,
        max_char_count=1000,
        sliding_window_overlap_chars=50,
    )

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.chunk_id == "1:2"
    assert chunk.clause_id == "1:2"
    assert chunk.source_clause_ids == ("1:2", "1:2.1")
    assert chunk.rule == ChunkRule.MERGED
    assert "2.1 Furto" in chunk.text
    assert "Curto." in chunk.text


@pytest.mark.unit
def test_fold_cascades_through_multiple_levels() -> None:
    grandchild = _clause(
        "1:3.1.1", parent_id="1:3.1", title="3.1.1 Item", content_lines=("Tiny.",)
    )
    child = _clause(
        "1:3.1",
        parent_id="1:3",
        child_ids=("1:3.1.1",),
        title="3.1 Sub",
        content_lines=("Also tiny.",),
    )
    root = _clause(
        "1:3",
        parent_id=None,
        child_ids=("1:3.1",),
        title="3. RISCOS EXCLUÍDOS",
        content_lines=("Ficam excluídos os seguintes riscos da presente apólice." * 3,),
    )

    chunks, _ = chunk_typed_clauses(
        [_typed(root), _typed(child), _typed(grandchild)],
        min_char_count=50,
        target_char_count=1000,
        max_char_count=1000,
        sliding_window_overlap_chars=50,
    )

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.clause_id == "1:3"
    assert chunk.source_clause_ids == ("1:3", "1:3.1", "1:3.1.1")
    assert chunk.rule == ChunkRule.MERGED


@pytest.mark.unit
def test_only_short_siblings_fold_long_sibling_stays_independent() -> None:
    short_child = _clause(
        "1:4.1", parent_id="1:4", title="4.1 Curto", content_lines=("x" * 10,)
    )
    long_child = _clause(
        "1:4.2",
        parent_id="1:4",
        title="4.2 Longo",
        content_lines=(
            "Texto suficientemente longo para permanecer como chunk independente." * 3,
        ),
    )
    root = _clause(
        "1:4", parent_id=None, child_ids=("1:4.1", "1:4.2"), title="4. GARANTIAS"
    )

    chunks, _ = chunk_typed_clauses(
        [_typed(root), _typed(short_child), _typed(long_child)],
        min_char_count=50,
        target_char_count=1000,
        max_char_count=1000,
        sliding_window_overlap_chars=50,
    )

    long_chunk = next(c for c in chunks if c.clause_id == "1:4.2")
    assert long_chunk.rule == ChunkRule.SINGLE
    assert long_chunk.source_clause_ids == ("1:4.2",)

    root_chunk = next(c for c in chunks if c.clause_id == "1:4")
    assert root_chunk.rule == ChunkRule.MERGED
    assert root_chunk.source_clause_ids == ("1:4", "1:4.1")


@pytest.mark.unit
def test_short_root_with_no_children_kept_as_is() -> None:
    clause = _clause(
        "1:5", parent_id=None, title="5. DEFINIÇÕES GERAIS", content_lines=("x",)
    )

    chunks, _ = chunk_typed_clauses(
        [_typed(clause)],
        min_char_count=50,
        target_char_count=1000,
        max_char_count=1000,
        sliding_window_overlap_chars=50,
    )

    assert len(chunks) == 1
    assert chunks[0].clause_id == "1:5"
    assert chunks[0].rule == ChunkRule.SINGLE


@pytest.mark.unit
def test_item_boundary_split_produces_multiple_pieces_same_clause_id() -> None:
    items = tuple(
        f"{chr(ord('a') + i)}) Risco excluído numero {i} descrito em detalhes."
        for i in range(10)
    )
    clause = _clause(
        "1:6", parent_id=None, title="6. RISCOS EXCLUÍDOS", content_lines=items
    )

    chunks, _ = chunk_typed_clauses(
        [_typed(clause)],
        min_char_count=1,
        target_char_count=200,
        max_char_count=250,
        sliding_window_overlap_chars=50,
    )

    assert len(chunks) > 1
    assert all(chunk.clause_id == "1:6" for chunk in chunks)
    assert all(chunk.rule == ChunkRule.ITEM_BOUNDARY_SPLIT for chunk in chunks)
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
    assert all(chunk.chunk_count == len(chunks) for chunk in chunks)
    assert all(chunk.source_clause_ids == ("1:6",) for chunk in chunks)
    for item in items:
        assert sum(item in chunk.text for chunk in chunks) == 1


@pytest.mark.unit
def test_sliding_window_used_when_no_item_boundaries() -> None:
    sentences = tuple(
        f"Esta e a sentenca numero {i} do texto continuo do glossario."
        for i in range(20)
    )
    clause = _clause(
        "1:7", parent_id=None, title="7. GLOSSÁRIO", content_lines=sentences
    )

    chunks, _ = chunk_typed_clauses(
        [_typed(clause)],
        min_char_count=1,
        target_char_count=200,
        max_char_count=250,
        sliding_window_overlap_chars=60,
    )

    assert len(chunks) > 1
    assert all(chunk.clause_id == "1:7" for chunk in chunks)
    assert all(chunk.rule == ChunkRule.SLIDING_WINDOW_SPLIT for chunk in chunks)
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
    for first, second in zip(chunks, chunks[1:], strict=False):
        shared = set(first.text.splitlines()) & set(second.text.splitlines())
        assert any(line in sentences for line in shared)


@pytest.mark.unit
def test_chunk_metadata_matches_source_typed_clause() -> None:
    clause = _clause(
        "1:8",
        parent_id=None,
        title="8. DEFINIÇÕES",
        content_lines=("Termo tecnico definido aqui.",),
        bundle_section="CASCO",
    )
    provenance = _provenance()
    typed = TypedClause(
        clause=clause,
        clause_type=ClauseType.DEFINITION,
        type_source=TypeSource.LLM,
        confidence=0.87,
        provenance=provenance,
    )

    chunks, _ = chunk_typed_clauses(
        [typed],
        min_char_count=1,
        target_char_count=1000,
        max_char_count=1000,
        sliding_window_overlap_chars=50,
    )

    chunk = chunks[0]
    assert chunk.clause_type == ClauseType.DEFINITION
    assert chunk.type_source == TypeSource.LLM
    assert chunk.confidence == 0.87
    assert chunk.bundle_section == "CASCO"
    assert chunk.provenance == provenance


@pytest.mark.unit
def test_report_counts_reconcile_with_emitted_chunks() -> None:
    short_child = _clause(
        "1:9.1", parent_id="1:9", title="9.1 Curto", content_lines=("x" * 5,)
    )
    parent = _clause(
        "1:9",
        parent_id=None,
        child_ids=("1:9.1",),
        title="9. COBERTURAS",
        content_lines=("Texto de cobertura razoavelmente longo para o teste." * 2,),
    )
    other = _clause(
        "1:10",
        parent_id=None,
        title="10. OUTRAS DISPOSIÇÕES",
        content_lines=(
            "Outro texto independente e de tamanho razoavel para o teste." * 2,
        ),
    )

    chunks, report = chunk_typed_clauses(
        [_typed(parent), _typed(short_child), _typed(other)],
        min_char_count=20,
        target_char_count=1000,
        max_char_count=1000,
        sliding_window_overlap_chars=50,
    )

    assert report.chunk_count == len(chunks)
    assert report.single_count == sum(1 for c in chunks if c.rule == ChunkRule.SINGLE)
    assert report.merged_count == sum(1 for c in chunks if c.rule == ChunkRule.MERGED)
    assert report.item_boundary_split_count == sum(
        1 for c in chunks if c.rule == ChunkRule.ITEM_BOUNDARY_SPLIT
    )
    assert report.sliding_window_split_count == sum(
        1 for c in chunks if c.rule == ChunkRule.SLIDING_WINDOW_SPLIT
    )
    assert report.max_char_count == max(c.char_count for c in chunks)
    assert report.min_char_count == min(c.char_count for c in chunks)


@pytest.mark.unit
def test_empty_input_returns_no_chunks() -> None:
    chunks, report = chunk_typed_clauses(
        [],
        min_char_count=1,
        target_char_count=1000,
        max_char_count=1000,
        sliding_window_overlap_chars=50,
    )

    assert chunks == []
    assert report.chunk_count == 0
