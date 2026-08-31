"""Unit tests for the embedding input-length check -- [M3-02].

No live model: the tokeniser is faked (duck-typed ``.encode(str).ids``),
matching the fake-chat-model precedent in the LLM-classifier tests.
"""

from dataclasses import dataclass

import pytest
from scripts.check_embedding_input_length import (
    render_summary,
    summarise_token_counts,
    token_counts,
)

from domain.chunk import Chunk, ChunkRule
from domain.clause_classification import ClauseProvenance, ClauseType, TypeSource
from infrastructure.rag.chunk_schema import ChunkRecord, flatten_chunk


@dataclass
class _FakeEncoding:
    ids: list[int]


class _FakeTokenizer:
    """Records what it was asked to encode; one token per whitespace group."""

    def __init__(self) -> None:
        self.seen: list[str] = []

    def encode(self, sequence: str) -> _FakeEncoding:
        self.seen.append(sequence)
        return _FakeEncoding(ids=list(range(len(sequence.split()))))


def _record(text: str) -> ChunkRecord:
    chunk = Chunk(
        document_id="1",
        chunk_id="1:c",
        clause_id="1:c",
        source_clause_ids=("1:c",),
        chunk_index=0,
        chunk_count=1,
        parent_path="",
        text=text,
        char_count=len(text),
        rule=ChunkRule.SINGLE,
        clause_type=ClauseType.COVERAGE,
        type_source=TypeSource.RULE,
        confidence=None,
        bundle_section=None,
        provenance=ClauseProvenance(
            document_id="1",
            susep_process="15414610650202459",
            insurer="Porto Seguro",
            cnpj="61198164000160",
            product_line="CASCO",
            indemnity_regime="VMR",
            process_year="2024",
        ),
    )
    return flatten_chunk(chunk, source="text")


@pytest.mark.unit
def test_summarise_empty_corpus_fits_trivially() -> None:
    summary = summarise_token_counts([], limit=8192)

    assert summary.chunk_count == 0
    assert summary.maximum == 0
    assert summary.n_over_limit == 0
    assert summary.fits is True


@pytest.mark.unit
def test_summarise_reports_the_distribution() -> None:
    summary = summarise_token_counts([10, 20, 30, 40, 50], limit=8192)

    assert summary.chunk_count == 5
    assert summary.minimum == 10
    assert summary.maximum == 50
    assert summary.p50 == 30
    assert summary.n_over_limit == 0
    assert summary.fits is True


@pytest.mark.unit
def test_summarise_flags_chunks_over_the_limit() -> None:
    summary = summarise_token_counts([100, 9000, 200, 8193], limit=8192)

    assert summary.n_over_limit == 2
    assert summary.fits is False
    assert summary.maximum == 9000


@pytest.mark.unit
def test_token_counts_tokenises_the_embedded_text_via_format_passage() -> None:
    tokenizer = _FakeTokenizer()
    records = [_record("uma duas tres"), _record("quatro")]

    counts = token_counts(records, tokenizer)

    assert counts == [3, 1]
    # format_passage is identity for the current model, so the tokeniser sees
    # exactly each chunk's embedded text.
    assert tokenizer.seen == ["uma duas tres", "quatro"]


@pytest.mark.unit
def test_render_summary_says_pass_when_everything_fits() -> None:
    text = render_summary(summarise_token_counts([1, 2, 3], limit=8192))

    assert "PASS" in text
    assert "FAIL" not in text


@pytest.mark.unit
def test_render_summary_says_fail_and_points_at_m3_01() -> None:
    text = render_summary(summarise_token_counts([9000], limit=8192))

    assert "FAIL" in text
    assert "M3-01" in text
