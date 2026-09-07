"""The Claim entity [M5-01]."""

import dataclasses
from datetime import UTC, datetime

import pytest

from domain.claim import Claim
from domain.susep_process import SusepProcess


def _claim(**overrides: object) -> Claim:
    fields: dict[str, object] = {
        "claim_id": "claim-1",
        "raw_text": "Bati o carro na traseira de outro veiculo ontem a noite.",
        "submitted_at": datetime(2026, 9, 3, 9, 30, tzinfo=UTC),
        "policy_ref": None,
    }
    fields.update(overrides)
    return Claim(**fields)  # type: ignore[arg-type]


@pytest.mark.unit
def test_accepts_a_minimal_claim() -> None:
    claim = _claim()

    assert claim.policy_ref is None


@pytest.mark.unit
def test_accepts_a_known_policy_reference() -> None:
    process = SusepProcess("15414.610650/2024-59")
    claim = _claim(policy_ref=process)

    assert claim.policy_ref == process


@pytest.mark.unit
@pytest.mark.parametrize("field", ["claim_id", "raw_text"])
def test_rejects_an_empty_string_field(field: str) -> None:
    with pytest.raises(ValueError):
        _claim(**{field: ""})


@pytest.mark.unit
def test_rejects_a_naive_submitted_at() -> None:
    with pytest.raises(ValueError):
        _claim(submitted_at=datetime(2026, 1, 1))  # noqa: DTZ001


@pytest.mark.unit
def test_is_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        _claim().raw_text = "x"  # type: ignore[misc]
