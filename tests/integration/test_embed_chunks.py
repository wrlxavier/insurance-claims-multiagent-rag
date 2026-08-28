"""scripts/embed_chunks.py's pipeline seam against a real Postgres -- [M3-02].

``run_embed_chunks`` + ``build_report`` over a real ``chunk`` table, driven by a
deterministic fake embedder -- no model, no optional ``embed`` group. The real
cold run (and its wall-clock number) is a manual step.
"""

import hashlib
import math
from collections.abc import Sequence
from pathlib import Path

import pytest
from scripts.embed_chunks import build_report, run_embed_chunks
from sqlalchemy.orm import Session

from infrastructure.database.chunk_repository import upsert_chunks
from infrastructure.rag.chunk_schema import SCHEMA_VERSION, ChunkRecord
from infrastructure.rag.embedding_cache import CachingEmbedder
from infrastructure.rag.embedding_config import EMBEDDING_DIMENSIONS

pytestmark = pytest.mark.integration

_VERSIONS = {"pgvector_version": "pgvector 0.8.6", "postgres_version": "PostgreSQL 17"}


def _unit_vector(text: str) -> list[float]:
    digest = hashlib.sha256(text.encode()).digest()
    raw = [digest[i % len(digest)] + 1 for i in range(EMBEDDING_DIMENSIONS)]
    norm = math.sqrt(sum(value * value for value in raw))
    return [value / norm for value in raw]


class FakeEmbedder:
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [_unit_vector(text) for text in texts]


def _record(chunk_id: str) -> ChunkRecord:
    return ChunkRecord.model_validate(
        {
            "schema_version": SCHEMA_VERSION,
            "chunk_id": chunk_id,
            "document_id": "1",
            "clause_id": chunk_id,
            "source_clause_ids": [chunk_id],
            "chunk_index": 0,
            "chunk_count": 1,
            "parent_path": "",
            "text": f"cláusula {chunk_id}",
            "display_text": f"cláusula {chunk_id}",
            "char_count": 12,
            "rule": "single",
            "clause_type": "coverage",
            "type_source": "rule",
            "confidence": None,
            "bundle_section": None,
            "source": "text",
            "susep_process": "15414.900666/2014-89",
            "insurer": "Bradesco Seguros",
            "cnpj": "12345678000199",
            "product_line": "CASCO",
            "indemnity_regime": "VD",
            "filing_year": "2019",
        }
    )


def test_run_embed_chunks_fills_the_column_then_re_runs_clean(
    db_session: Session, tmp_path: Path
) -> None:
    upsert_chunks(db_session, [_record(f"1:{i}") for i in range(5)])
    db_session.commit()

    embedder = CachingEmbedder(FakeEmbedder(), cache_path=tmp_path / "cache.jsonl")
    run, wall = run_embed_chunks(db_session, embedder, batch_size=2)

    assert run.embedded == 5
    assert run.already_present == 0
    assert run.batches == 3
    assert (embedder.hits, embedder.misses) == (0, 5)
    assert wall >= 0.0

    report = build_report(
        run=run,
        wall_clock_seconds=wall,
        token_counts=[100] * 5,
        chunk_count=5,
        cache_hits=embedder.hits,
        cache_misses=embedder.misses,
        batch_size=2,
        device="cpu",
        versions=_VERSIONS,
    )
    assert report.dollar_cost_usd == 0.0
    assert report.chunks_embedded == 5
    assert report.cache_misses == 5

    second, _ = run_embed_chunks(db_session, FakeEmbedder(), batch_size=2)
    assert second.embedded == 0
    assert second.already_present == 5
