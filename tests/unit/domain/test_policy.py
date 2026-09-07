"""The Policy entity [M5-01]."""

import dataclasses
from pathlib import Path

import pytest

from domain.cnpj import Cnpj
from domain.errors import InvalidValueObjectError, PolicyYearMismatchError
from domain.policy import Policy
from domain.susep_process import SusepProcess
from infrastructure.parsing.manifest import read_manifest

_MANIFEST = read_manifest(Path("data/policies/manifest.csv"))
_ROW_1 = _MANIFEST[0]
_ROW_26 = next(row for row in _MANIFEST if row["id"] == "26")


def _policy(**overrides: object) -> Policy:
    fields: dict[str, object] = {
        "susep_process": SusepProcess("15414.610650/2024-59"),
        "cnpj": Cnpj("61198164000160"),
        "insurer": "PORTO SEGURO COMPANHIA DE SEGUROS GERAIS",
        "product_line": "CASCO",
        "indemnity_regime": "VD",
        "process_year": "2024",
    }
    fields.update(overrides)
    return Policy(**fields)  # type: ignore[arg-type]


@pytest.mark.unit
def test_accepts_a_full_policy() -> None:
    policy = _policy()

    assert policy.identity == SusepProcess("15414.610650/2024-59")
    assert isinstance(policy.identity, SusepProcess)


@pytest.mark.unit
def test_from_manifest_row_applies_the_cnpj_zero_pad() -> None:
    policy = Policy.from_manifest_row(_ROW_26)

    assert policy.cnpj.value == "09180505000150"
    assert len(policy.cnpj.value) == 14


@pytest.mark.unit
def test_from_manifest_row_round_trips_every_corpus_row() -> None:
    for row in _MANIFEST:
        policy = Policy.from_manifest_row(row)
        assert policy.susep_process.value == row["susep_process"]
        assert policy.process_year == row["process_year"]


@pytest.mark.unit
def test_rejects_a_year_that_disagrees_with_the_process() -> None:
    with pytest.raises(PolicyYearMismatchError):
        _policy(process_year="2023")


@pytest.mark.unit
@pytest.mark.parametrize("field", ["insurer", "product_line", "indemnity_regime"])
def test_rejects_an_empty_descriptive_field(field: str) -> None:
    with pytest.raises(ValueError):
        _policy(**{field: ""})


@pytest.mark.unit
def test_from_manifest_row_surfaces_a_malformed_identifier() -> None:
    bad = {**_ROW_1, "cnpj": "not-a-cnpj"}
    with pytest.raises(InvalidValueObjectError):
        Policy.from_manifest_row(bad)


@pytest.mark.unit
def test_is_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        _policy().insurer = "x"  # type: ignore[misc]
