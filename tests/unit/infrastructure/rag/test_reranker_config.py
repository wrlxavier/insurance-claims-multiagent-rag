"""The pinned cross-encoder-reranker contract and its anti-drift guards -- [M3-05]."""

import re
from pathlib import Path

import pytest

from infrastructure.rag import reranker_config
from infrastructure.rag.reranker_config import (
    RERANK_CANDIDATE_DEPTH,
    RERANKER_MAX_INPUT_TOKENS,
    RERANKER_MODEL_ID,
    RERANKER_MODEL_REVISION,
    RERANKER_TRUST_REMOTE_CODE,
    config_fingerprint,
)

_REPO_ROOT = Path(__file__).resolve().parents[4]


@pytest.mark.unit
def test_model_contract_constants() -> None:
    assert RERANKER_MODEL_ID == "Alibaba-NLP/gte-multilingual-reranker-base"
    assert RERANKER_MAX_INPUT_TOKENS == 8192
    assert RERANKER_TRUST_REMOTE_CODE is True
    assert RERANK_CANDIDATE_DEPTH >= 10


@pytest.mark.unit
def test_revision_is_a_pinned_commit_not_a_floating_alias() -> None:
    # A 40-char hex SHA -- never "main"/"master"/"" -- so a provider-side
    # re-upload cannot change the ranking behind a published number.
    assert re.fullmatch(r"[0-9a-f]{40}", RERANKER_MODEL_REVISION) is not None


@pytest.mark.unit
def test_config_fingerprint_is_stable() -> None:
    # A pinned literal, so reordering or dropping a field from the payload is a
    # visible test failure -- the reranker cache's key depends on this digest.
    assert config_fingerprint() == "777c0503f1073d52"
    assert re.fullmatch(r"[0-9a-f]{16}", config_fingerprint()) is not None


@pytest.mark.unit
@pytest.mark.parametrize(
    ("attr", "value"),
    [
        ("RERANKER_MODEL_ID", "other/model"),
        ("RERANKER_MODEL_REVISION", "0" * 40),
        ("RERANKER_MAX_INPUT_TOKENS", 512),
        ("RERANK_CANDIDATE_DEPTH", 999),
    ],
)
def test_every_contract_field_changes_the_fingerprint(
    monkeypatch: pytest.MonkeyPatch, attr: str, value: object
) -> None:
    baseline = config_fingerprint()
    monkeypatch.setattr(reranker_config, attr, value)
    assert config_fingerprint() != baseline


@pytest.mark.unit
def test_env_example_reranker_model_matches_the_pinned_id() -> None:
    env_example = (_REPO_ROOT / ".env.example").read_text(encoding="utf-8")
    values = {
        key.strip(): value.strip().strip('"').strip("'")
        for line in env_example.splitlines()
        if line.strip() and not line.lstrip().startswith("#") and "=" in line
        for key, _, value in [line.partition("=")]
    }
    assert values["RERANKER_MODEL"] == RERANKER_MODEL_ID
