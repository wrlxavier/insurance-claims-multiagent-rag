"""The metadata pre-filter -- [M3-04]."""

import pytest

from domain.clause_classification import ClauseType
from infrastructure.rag.chunk_schema import ChunkRecord
from infrastructure.rag.retrieval_filter import RetrievalFilter


def _chunk(
    *,
    susep_process: str = "15414.610650/2024-59",
    cnpj: str = "61198164000160",
    product_line: str = "CASCO",
    bundle_section: str | None = None,
    clause_type: str = "coverage",
) -> ChunkRecord:
    return ChunkRecord.model_validate(
        {
            "schema_version": "v1",
            "chunk_id": "1:c",
            "document_id": "1",
            "clause_id": "1:c",
            "source_clause_ids": ["1:c"],
            "chunk_index": 0,
            "chunk_count": 1,
            "parent_path": "",
            "text": "texto",
            "display_text": "texto",
            "char_count": 5,
            "rule": "single",
            "clause_type": clause_type,
            "type_source": "rule",
            "confidence": None,
            "bundle_section": bundle_section,
            "source": "text",
            "susep_process": susep_process,
            "insurer": "Seguradora",
            "cnpj": cnpj,
            "product_line": product_line,
            "indemnity_regime": "VD",
            "filing_year": "2024",
        }
    )


@pytest.mark.unit
def test_from_manifest_row_builds_the_default_process_cnpj_filter() -> None:
    row = {"susep_process": "15414.610650/2024-59", "cnpj": "61198164000160"}
    filt = RetrievalFilter.from_manifest_row(row)

    assert filt.susep_process == "15414.610650/2024-59"
    assert filt.cnpj == "61198164000160"
    assert filt.product_line is None
    assert not filt.is_empty


@pytest.mark.unit
def test_all_none_filter_is_empty_and_matches_everything() -> None:
    filt = RetrievalFilter()

    assert filt.is_empty
    assert filt.matches(_chunk())


@pytest.mark.unit
def test_process_and_cnpj_equality() -> None:
    filt = RetrievalFilter(susep_process="A", cnpj="X")

    assert filt.matches(_chunk(susep_process="A", cnpj="X"))
    assert not filt.matches(_chunk(susep_process="B", cnpj="X"))
    assert not filt.matches(_chunk(susep_process="A", cnpj="Y"))


@pytest.mark.unit
def test_clause_type_is_compared_as_the_enum() -> None:
    filt = RetrievalFilter(clause_type=ClauseType.EXCLUSION)

    assert filt.matches(_chunk(clause_type="exclusion"))
    assert not filt.matches(_chunk(clause_type="coverage"))


@pytest.mark.unit
def test_bundle_section_lenient_default_keeps_null_chunks() -> None:
    filt = RetrievalFilter(bundle_section="Motocicletas")

    assert filt.matches(_chunk(bundle_section="Motocicletas"))
    assert filt.matches(_chunk(bundle_section=None))
    assert not filt.matches(_chunk(bundle_section="Veículos de Carga"))


@pytest.mark.unit
def test_bundle_section_strict_excludes_null_chunks() -> None:
    filt = RetrievalFilter(bundle_section="Motocicletas", strict_bundle=True)

    assert filt.matches(_chunk(bundle_section="Motocicletas"))
    assert not filt.matches(_chunk(bundle_section=None))


@pytest.mark.unit
def test_strict_bundle_is_inert_without_a_bundle_section() -> None:
    filt = RetrievalFilter(cnpj="X", strict_bundle=True)

    assert filt.matches(_chunk(cnpj="X", bundle_section=None))
