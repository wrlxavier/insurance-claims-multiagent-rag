"""The pinned insufficient-context gate contract and its anti-drift guard -- [M3-07]."""

import re

import pytest

from infrastructure.rag import insufficient_context_config as config
from infrastructure.rag.insufficient_context_config import (
    INSTANCE_VALUE_TOP_SCORE_THRESHOLD,
    TOP_SCORE_ABSTAIN_THRESHOLD,
    config_fingerprint,
)


@pytest.mark.unit
def test_contract_constants() -> None:
    # A sigmoid rerank score in [0, 1]; a threshold outside that range would
    # make a floor always-on or always-off. The strict floor is above the loose
    # one by construction.
    assert 0.0 < TOP_SCORE_ABSTAIN_THRESHOLD < 1.0
    assert TOP_SCORE_ABSTAIN_THRESHOLD < INSTANCE_VALUE_TOP_SCORE_THRESHOLD < 1.0


@pytest.mark.unit
def test_config_fingerprint_is_stable() -> None:
    # A pinned literal: changing a threshold without re-running the calibration
    # (and updating docs/INSUFFICIENT_CONTEXT_GATE.md) is a visible test failure.
    assert config_fingerprint() == "c29b8ebee67be01b"
    assert re.fullmatch(r"[0-9a-f]{16}", config_fingerprint()) is not None


@pytest.mark.unit
@pytest.mark.parametrize(
    "attr", ["TOP_SCORE_ABSTAIN_THRESHOLD", "INSTANCE_VALUE_TOP_SCORE_THRESHOLD"]
)
def test_every_threshold_moves_the_fingerprint(
    monkeypatch: pytest.MonkeyPatch, attr: str
) -> None:
    baseline = config_fingerprint()
    monkeypatch.setattr(config, attr, 0.999)
    assert config_fingerprint() != baseline
