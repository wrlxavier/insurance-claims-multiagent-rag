"""The demo helper API -- ``/demo/api/*`` [M6-03].

The demo router has no ``Depends`` and no ``app.state`` coupling, so a bare
``TestClient(create_app())`` (no lifespan) exercises it -- no Postgres, no graph,
no LLM. What is worth testing is the real logic: the curated examples stay
verbatim copies of the synthetic set, the clause-context join, and its graceful
degradation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from presentation.app import create_app
from presentation.demo import clause_context
from presentation.demo.examples import load_examples
from presentation.demo.schemas import ClauseContext

_SYNTHETIC_FILES = (
    Path("data/synthetic_claims/claims.jsonl"),
    Path("data/synthetic_claims/product_claim_mismatch.jsonl"),
)
_CATEGORIES = {
    "compatible",
    "incompatible",
    "insufficient_information",
    "product_claim_mismatch",
}


@pytest.fixture
def client() -> TestClient:
    """A client over the app without its lifespan -- the demo router needs none."""
    return TestClient(create_app())


@pytest.fixture(autouse=True)
def _clear_caches() -> None:
    clause_context.clause_index.cache_clear()
    load_examples.cache_clear()


def _synthetic_narratives() -> dict[str, str]:
    out: dict[str, str] = {}
    for path in _SYNTHETIC_FILES:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                out[row["claim_id"]] = row["narrative"]
    return out


def test_examples_endpoint_returns_a_curated_covering_set(client: TestClient) -> None:
    body = client.get("/demo/api/examples").json()

    assert len(body) >= 6
    assert {row["category"] for row in body} == _CATEGORIES
    for row in body:
        assert row["expected_verdict"] in {
            "compatible",
            "incompatible",
            "insufficient_information",
        }
        assert row["raw_text"].strip()


def test_examples_are_verbatim_synthetic_narratives(client: TestClient) -> None:
    narratives = _synthetic_narratives()

    for row in client.get("/demo/api/examples").json():
        assert row["source_claim_id"] in narratives, row["source_claim_id"]
        assert row["raw_text"] == narratives[row["source_claim_id"]], row["id"]


def test_clause_context_joins_the_full_clause(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = ClauseContext(
        clause_id="1:glossario",
        document_id="1",
        title="GLOSSARIO",
        text="Aceitacao: e a aprovacao da proposta.",
        page_start=4,
        page_end=7,
        clause_type="definition",
        susep_process="15414.610650/2024-59",
        insurer="PORTO SEGURO",
        product_line="CASCO",
        filing_year="2024",
    )
    from presentation.demo import router

    monkeypatch.setattr(router, "clause_index", lambda: {"1:glossario": fake})

    body = client.get(
        "/demo/api/clause-context",
        params={"clause_id": ["1:glossario", "ghost:9"]},
    ).json()

    assert body["index_available"] is True
    assert body["contexts"]["1:glossario"]["page_start"] == 4
    assert body["contexts"]["1:glossario"]["text"].startswith("Aceitacao")
    assert body["missing"] == ["ghost:9"]


def test_clause_context_degrades_without_the_corpus(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from infrastructure.parsing import corpus_artifact

    monkeypatch.setattr(
        corpus_artifact, "JSONL_PATH", tmp_path / "missing.jsonl", raising=True
    )
    clause_context.clause_index.cache_clear()

    body = client.get(
        "/demo/api/clause-context", params={"clause_id": ["1:glossario"]}
    ).json()

    assert body["index_available"] is False
    assert body["contexts"] == {}
    assert body["missing"] == ["1:glossario"]


def test_clause_context_requires_at_least_one_id(client: TestClient) -> None:
    assert client.get("/demo/api/clause-context").status_code == 422


def test_progress_reports_unavailable_when_the_checkpoint_cannot_be_read(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from infrastructure.graph import checkpointer

    def _boom(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("no checkpointer here")

    monkeypatch.setattr(checkpointer, "open_claim_checkpointer", _boom)

    body = client.get("/demo/api/assessments/whatever/progress").json()

    assert body["available"] is False
    assert body["started"] is False
    assert body["pipeline"][0] == "intake"
    assert body["nodes_seen"] == []
