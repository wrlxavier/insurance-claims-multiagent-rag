"""BM25 lexical retriever + chunk->clause roll-up -- [M3-03]."""

import pytest

from infrastructure.rag.chunk_schema import ChunkRecord
from infrastructure.rag.lexical_retriever import LexicalRetriever


class WhitespaceAnalyzer:
    """A fake analyzer: lowercase + whitespace split, no stemmer, no I/O."""

    def analyze(self, text: str) -> list[str]:
        return text.lower().split()


def _chunk(
    chunk_id: str,
    clause_id: str,
    source_clause_ids: list[str],
    text: str,
) -> ChunkRecord:
    return ChunkRecord.model_validate(
        {
            "schema_version": "v1",
            "chunk_id": chunk_id,
            "document_id": "1",
            "clause_id": clause_id,
            "source_clause_ids": source_clause_ids,
            "chunk_index": 0,
            "chunk_count": 1,
            "parent_path": "",
            "text": text,
            "display_text": text,
            "char_count": len(text),
            "rule": "single",
            "clause_type": "coverage",
            "type_source": "rule",
            "confidence": 1.0,
            "bundle_section": None,
            "source": "text",
            "susep_process": "123",
            "insurer": "Insurer",
            "cnpj": "123",
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
