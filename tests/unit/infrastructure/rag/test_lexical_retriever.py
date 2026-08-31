"""BM25 lexical retriever + chunk->clause roll-up -- [M3-03]."""

import pytest

from domain.clause_classification import ClauseType
from infrastructure.rag.chunk_schema import ChunkRecord
from infrastructure.rag.lexical_retriever import LexicalRetriever
from infrastructure.rag.retrieval_filter import RetrievalFilter


class WhitespaceAnalyzer:
    """A fake analyzer: lowercase + whitespace split, no stemmer, no I/O."""

    def analyze(self, text: str) -> list[str]:
        return text.lower().split()


def _chunk(
    chunk_id: str,
    clause_id: str,
    source_clause_ids: list[str],
    text: str,
    *,
    document_id: str = "1",
    susep_process: str = "123",
    cnpj: str = "123",
    clause_type: str = "coverage",
) -> ChunkRecord:
    return ChunkRecord.model_validate(
        {
            "schema_version": "v1",
            "chunk_id": chunk_id,
            "document_id": document_id,
            "clause_id": clause_id,
            "source_clause_ids": source_clause_ids,
            "chunk_index": 0,
            "chunk_count": 1,
            "parent_path": "",
            "text": text,
            "display_text": text,
            "char_count": len(text),
            "rule": "single",
            "clause_type": clause_type,
            "type_source": "rule",
            "confidence": 1.0,
            "bundle_section": None,
            "source": "text",
            "susep_process": susep_process,
            "insurer": "Insurer",
            "cnpj": cnpj,
            "product_line": "CASCO",
            "indemnity_regime": "VD",
            "filing_year": "2020",
        }
    )


_CHUNKS = [
    _chunk("1:cov", "1:cov", ["1:cov"], "franquia vidros cobertura"),
    # A short clause "1:merged" folded into its parent: it is in no chunk's
    # `clause_id`, only in this chunk's `source_clause_ids`.
    _chunk("1:parent", "1:parent", ["1:parent", "1:merged"], "exclusao guerra"),
    # One over-long clause split into two chunks, both anchored on "1:split".
    _chunk("1:split#0", "1:split", ["1:split"], "reboque guincho assistencia"),
    _chunk("1:split#1", "1:split", ["1:split"], "reboque limite quilometros"),
]


def _retriever() -> LexicalRetriever:
    return LexicalRetriever.from_chunks(_CHUNKS, WhitespaceAnalyzer())


@pytest.mark.unit
def test_exact_term_ranks_the_owning_clause_first() -> None:
    assert _retriever().retrieve("franquia", k=10)[0] == "1:cov"


@pytest.mark.unit
def test_a_merged_clause_is_retrievable_via_source_clause_ids() -> None:
    hits = _retriever().retrieve("guerra", k=10)
    assert "1:parent" in hits and "1:merged" in hits


@pytest.mark.unit
def test_split_chunks_collapse_to_one_clause_id() -> None:
    hits = _retriever().retrieve("reboque", k=10)
    assert hits.count("1:split") == 1


@pytest.mark.unit
def test_returns_fewer_than_k_without_padding() -> None:
    assert _retriever().retrieve("franquia", k=10) == ["1:cov"]


@pytest.mark.unit
def test_non_positive_k_returns_empty() -> None:
    assert _retriever().retrieve("franquia", k=0) == []


@pytest.mark.unit
def test_retrieval_is_deterministic() -> None:
    retriever = _retriever()
    query = "reboque exclusao franquia"
    assert retriever.retrieve(query, k=5) == retriever.retrieve(query, k=5)


# -- [M3-04] metadata filter + retrieve_scored -------------------------------- #

_CROSS_DOC_CHUNKS = [
    _chunk("1:a", "1:a", ["1:a"], "franquia reduzida vidros", susep_process="P1"),
    _chunk("2:a", "2:a", ["2:a"], "franquia reduzida vidros", susep_process="P2"),
]


@pytest.mark.unit
def test_a_process_filter_keeps_only_the_matching_document() -> None:
    retriever = LexicalRetriever.from_chunks(_CROSS_DOC_CHUNKS, WhitespaceAnalyzer())
    filt = RetrievalFilter(susep_process="P1")

    assert retriever.retrieve("franquia", k=10, metadata_filter=filt) == ["1:a"]


@pytest.mark.unit
def test_a_clause_type_filter_drops_non_matching_chunks() -> None:
    chunks = [
        _chunk("1:cov", "1:cov", ["1:cov"], "guerra", clause_type="coverage"),
        _chunk("1:exc", "1:exc", ["1:exc"], "guerra", clause_type="exclusion"),
    ]
    retriever = LexicalRetriever.from_chunks(chunks, WhitespaceAnalyzer())
    filt = RetrievalFilter(clause_type=ClauseType.EXCLUSION)

    assert retriever.retrieve("guerra", k=10, metadata_filter=filt) == ["1:exc"]


@pytest.mark.unit
def test_retrieve_scored_returns_bm25_scores_best_first() -> None:
    scored = _retriever().retrieve_scored("franquia vidros cobertura", k=10)

    assert scored[0][0] == "1:cov"
    assert all(isinstance(score, float) for _, score in scored)
    assert [s for _, s in scored] == sorted((s for _, s in scored), reverse=True)


@pytest.mark.unit
def test_retrieve_projects_retrieve_scored() -> None:
    retriever = _retriever()
    query = "reboque exclusao franquia"
    assert retriever.retrieve(query, k=5) == [
        clause_id for clause_id, _ in retriever.retrieve_scored(query, k=5)
    ]
