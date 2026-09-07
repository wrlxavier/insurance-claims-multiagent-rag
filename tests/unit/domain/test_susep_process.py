"""The SusepProcess value object [M5-01]."""

import dataclasses
from pathlib import Path

import pytest

from domain.errors import InvalidSusepProcessError
from domain.susep_process import SusepProcess
from infrastructure.parsing.manifest import read_manifest

_CANONICAL = "15414.610650/2024-59"
_STEM = "15414610650202459"


@pytest.mark.unit
def test_accepts_the_canonical_written_form() -> None:
    process = SusepProcess(_CANONICAL)

    assert process.value == _CANONICAL


@pytest.mark.unit
def test_parse_accepts_the_seventeen_digit_filename_stem() -> None:
    assert SusepProcess.parse(_STEM) == SusepProcess(_CANONICAL)


@pytest.mark.unit
def test_parse_strips_surrounding_whitespace() -> None:
    assert SusepProcess.parse(f"  {_CANONICAL}\n") == SusepProcess(_CANONICAL)


@pytest.mark.unit
def test_decomposition_properties() -> None:
    process = SusepProcess(_CANONICAL)

    assert process.digits == _STEM
    assert process.filename_stem == _STEM
    assert process.year == "2024"


@pytest.mark.unit
def test_filename_stem_matches_the_manifest_filename() -> None:
    rows = read_manifest(Path("data/policies/manifest.csv"))
    for row in rows:
        process = SusepProcess.parse(row["susep_process"])
        assert f"{process.filename_stem}.pdf" == row["filename"]
        assert process.year == row["process_year"]


@pytest.mark.unit
@pytest.mark.parametrize(
    "raw",
    [
        "",
        "15414.610650/2024",  # no check group
        "1541.610650/2024-59",  # 4-digit prefix
        "15414-610650/2024-59",  # wrong separator
        "15414.610650/2024-5",  # 1-digit check group
        "1541X.610650/2024-59",  # letter in the number
        "15414.610650/2024-59 ",  # trailing space, bare constructor
        " 15414.610650/2024-59",  # leading space, bare constructor
    ],
)
def test_constructor_rejects_a_malformed_value(raw: str) -> None:
    with pytest.raises(InvalidSusepProcessError):
        SusepProcess(raw)


@pytest.mark.unit
@pytest.mark.parametrize("raw", ["1541461065020245", "154146106502024590"])
def test_parse_rejects_a_wrong_length_digit_string(raw: str) -> None:
    with pytest.raises(InvalidSusepProcessError):
        SusepProcess.parse(raw)


@pytest.mark.unit
def test_is_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        SusepProcess(_CANONICAL).value = "x"  # type: ignore[misc]
