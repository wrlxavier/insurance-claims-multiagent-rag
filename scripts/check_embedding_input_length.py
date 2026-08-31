#!/usr/bin/env python3
"""Confirm [M3-01]'s chunk lengths fit the embedding model's input window -- [M3-02].

The [M3-02] DoD requires recording the pinned model's maximum input length
and confirming that [M3-01]'s chunk-length rules fit inside it -- and, if
they do not, reporting the constraint back to [M3-01] rather than silently
truncating. This script does the check with the real tokenizer instead of a
chars-per-token estimate: it loads
[infrastructure.rag.embedding_config.EMBEDDING_MODEL_ID] at its pinned
revision, tokenises every chunk in ``build/chunks.jsonl`` exactly as it would
be embedded (via ``format_passage``, special tokens included), and reports
the token-count distribution against
[infrastructure.rag.embedding_config.EMBEDDING_MAX_INPUT_TOKENS].

Exits non-zero if any chunk exceeds the limit -- that is the "report back to
[M3-01]" signal, not something to paste past. Run via
``make check-embedding-input-length`` after ``make build-chunks`` (or
``make fetch-corpus-artifacts``). Downloads ``tokenizer.json`` once, pinned
to the model revision; no model weights, no torch.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from tokenizers import Tokenizer

from infrastructure.rag.chunk_artifact import CHUNKS_JSONL_PATH, read_chunks_jsonl
from infrastructure.rag.chunk_schema import ChunkRecord
from infrastructure.rag.embedding_config import (
    EMBEDDING_MAX_INPUT_TOKENS,
    EMBEDDING_MODEL_ID,
    EMBEDDING_MODEL_REVISION,
    format_passage,
)


class _Encoding(Protocol):
    @property
    def ids(self) -> list[int]: ...


class _Tokenizer(Protocol):
    def encode(self, sequence: str) -> _Encoding: ...


@dataclass(frozen=True)
class TokenLengthSummary:
    """The chunk token-length distribution against a model's input limit."""

    chunk_count: int
    limit: int
    minimum: int
    p50: int
    p90: int
    p99: int
    maximum: int
    n_over_limit: int

    @property
    def fits(self) -> bool:
        """True when no chunk would be truncated by the model."""
        return self.n_over_limit == 0


def _percentile(sorted_values: list[int], fraction: float) -> int:
    """Nearest-rank percentile, same formula as ``chunking._build_report``."""
    index = min(int(len(sorted_values) * fraction), len(sorted_values) - 1)
    return sorted_values[index]


def token_counts(records: list[ChunkRecord], tokenizer: _Tokenizer) -> list[int]:
    """Token count per chunk, tokenising exactly what the index side embeds."""
    return [
        len(tokenizer.encode(format_passage(record.text)).ids) for record in records
    ]


def summarise_token_counts(counts: list[int], *, limit: int) -> TokenLengthSummary:
    """Summarise per-chunk token counts against ``limit``. Never raises on empty."""
    if not counts:
        return TokenLengthSummary(0, limit, 0, 0, 0, 0, 0, 0)
    ordered = sorted(counts)
    return TokenLengthSummary(
        chunk_count=len(ordered),
        limit=limit,
        minimum=ordered[0],
        p50=_percentile(ordered, 0.50),
        p90=_percentile(ordered, 0.90),
        p99=_percentile(ordered, 0.99),
        maximum=ordered[-1],
        n_over_limit=sum(1 for count in ordered if count > limit),
    )


def render_summary(summary: TokenLengthSummary) -> str:
    """Render the summary as a short Markdown block for stdout and the PR."""
    lines = [
        f"# Embedding input length -- {EMBEDDING_MODEL_ID}",
        "",
        f"Revision `{EMBEDDING_MODEL_REVISION}`; limit "
        f"**{summary.limit}** tokens (incl. special tokens).",
        "",
        "| chunks | min | p50 | p90 | p99 | max | over limit |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| {summary.chunk_count} | {summary.minimum} | {summary.p50} "
        f"| {summary.p90} | {summary.p99} | {summary.maximum} "
        f"| {summary.n_over_limit} |",
        "",
    ]
    if summary.fits:
        lines.append(
            f"PASS -- every chunk fits ({summary.maximum}/{summary.limit} tokens at "
            "the tail). [M3-01]'s chunk-length rules need no change."
        )
    else:
        lines.append(
            f"FAIL -- {summary.n_over_limit} chunk(s) exceed {summary.limit} tokens "
            "and would be truncated. Report this back to [M3-01] (lower "
            "`CHUNK_MAX_CHAR_COUNT`) rather than embedding truncated text."
        )
    return "\n".join(lines)


def load_records(path: Path) -> list[ChunkRecord]:
    """Load the chunk corpus, failing loudly if it has not been built."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Run `make build-chunks` (full rebuild) or "
            "`make fetch-corpus-artifacts` (pre-built corpus) first."
        )
    return read_chunks_jsonl(path)


def main() -> None:
    """Tokenise the built chunk corpus and check it against the model's limit."""
    records = load_records(CHUNKS_JSONL_PATH)
    tokenizer = Tokenizer.from_pretrained(
        EMBEDDING_MODEL_ID, revision=EMBEDDING_MODEL_REVISION
    )
    summary = summarise_token_counts(
        token_counts(records, tokenizer), limit=EMBEDDING_MAX_INPUT_TOKENS
    )
    print(render_summary(summary))
    if not summary.fits:
        sys.exit(1)


if __name__ == "__main__":
    main()
