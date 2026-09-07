"""The pinned classifier contract and its anti-drift guard -- [M5-08 Appendix]."""

import re
from pathlib import Path

import pytest

from infrastructure.guardrails.classifier_config import (
    CLASSIFIER_MAX_INPUT_TOKENS,
    CLASSIFIER_MODEL_ID,
    CLASSIFIER_MODEL_REVISION,
    CLASSIFIER_POSITIVE_LABEL,
)

_REPO_ROOT = Path(__file__).resolve().parents[4]


@pytest.mark.unit
def test_model_contract_constants() -> None:
    assert CLASSIFIER_MODEL_ID == "protectai/deberta-v3-base-prompt-injection-v2"
    assert CLASSIFIER_MAX_INPUT_TOKENS == 512
    assert CLASSIFIER_POSITIVE_LABEL == "INJECTION"


@pytest.mark.unit
def test_revision_is_a_pinned_commit_not_a_floating_alias() -> None:
    # A 40-char hex SHA -- never "main"/"master"/"" -- so a provider-side
    # re-upload cannot change the weights behind an already-published number.
    assert re.fullmatch(r"[0-9a-f]{40}", CLASSIFIER_MODEL_REVISION) is not None


@pytest.mark.unit
def test_env_example_classifier_model_matches_the_pinned_id() -> None:
    env_example = (_REPO_ROOT / ".env.example").read_text(encoding="utf-8")
    values = {
        key.strip(): value.strip().strip('"').strip("'")
        for line in env_example.splitlines()
        if line.strip() and not line.lstrip().startswith("#") and "=" in line
        for key, _, value in [line.partition("=")]
    }
    assert values["PROMPT_INJECTION_CLASSIFIER_MODEL"] == CLASSIFIER_MODEL_ID
