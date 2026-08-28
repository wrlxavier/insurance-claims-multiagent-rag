"""The pinned hybrid-retrieval contract and its anti-drift guards -- [M3-04]."""

import re

import pytest

from infrastructure.rag import hybrid_config
from infrastructure.rag.hybrid_config import (
    CANDIDATE_DEPTH,
    DEFAULT_FUSION_STRATEGY,
    FUSION_WEIGHTS,
    RRF_K,
    FusionStrategy,
    config_fingerprint,
)

_LEXICAL_FP = "ef0a2dd0c1dfb4e4"


@pytest.mark.unit
def test_contract_constants() -> None:
    assert RRF_K == 60
    assert FUSION_WEIGHTS == (0.5, 0.5)
    assert CANDIDATE_DEPTH == 100
    assert DEFAULT_FUSION_STRATEGY is FusionStrategy.RRF


@pytest.mark.unit
def test_config_fingerprint_shape_and_dependence_on_the_lexical_fingerprint() -> None:
    digest = config_fingerprint(lexical_config_fingerprint=_LEXICAL_FP)
    assert re.fullmatch(r"[0-9a-f]{16}", digest)
    assert config_fingerprint(lexical_config_fingerprint="other") != digest


@pytest.mark.unit
@pytest.mark.parametrize(
    ("attr", "value"),
    [
        ("RRF_K", 40),
        ("FUSION_WEIGHTS", (0.7, 0.3)),
        ("CANDIDATE_DEPTH", 50),
        ("DEFAULT_FUSION_STRATEGY", FusionStrategy.WEIGHTED),
    ],
)
def test_every_contract_field_changes_the_fingerprint(
    monkeypatch: pytest.MonkeyPatch, attr: str, value: object
) -> None:
    baseline = config_fingerprint(lexical_config_fingerprint=_LEXICAL_FP)
    monkeypatch.setattr(hybrid_config, attr, value)
    assert config_fingerprint(lexical_config_fingerprint=_LEXICAL_FP) != baseline


@pytest.mark.unit
def test_fingerprint_moves_with_the_embedding_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from infrastructure.rag import embedding_config

    baseline = config_fingerprint(lexical_config_fingerprint=_LEXICAL_FP)
    monkeypatch.setattr(embedding_config, "EMBEDDING_MODEL_REVISION", "deadbeef")
    assert config_fingerprint(lexical_config_fingerprint=_LEXICAL_FP) != baseline
