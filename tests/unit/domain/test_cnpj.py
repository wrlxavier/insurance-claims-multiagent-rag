"""The Cnpj value object [M5-01]."""

import dataclasses
from pathlib import Path

import pytest

from domain.cnpj import Cnpj, _cnpj_check_digits
from domain.errors import InvalidCnpjError
from infrastructure.parsing.manifest import read_manifest

# Porto Seguro, manifest row 1 -- a real, valid CNPJ.
_VALID = "61198164000160"
_MANIFEST_CNPJS = [
    row["cnpj"] for row in read_manifest(Path("data/policies/manifest.csv"))
]


@pytest.mark.unit
def test_accepts_a_valid_fourteen_digit_cnpj() -> None:
    assert Cnpj(_VALID).value == _VALID


@pytest.mark.unit
def test_parse_left_pads_the_thirteen_digit_catalogue_defect() -> None:
    # SUSEP's open catalogue drops the leading zero; the corpus stores the
    # corrected 14-digit form (manifest row 26, USEBENS Seguros).
    assert Cnpj.parse("9180505000150") == Cnpj("09180505000150")


@pytest.mark.unit
def test_parse_strips_punctuation() -> None:
    assert Cnpj.parse("61.198.164/0001-60") == Cnpj(_VALID)


@pytest.mark.unit
def test_formatted_renders_the_punctuated_form() -> None:
    assert Cnpj(_VALID).formatted == "61.198.164/0001-60"


@pytest.mark.unit
@pytest.mark.parametrize(
    "raw",
    [
        "",
        "12345",  # too short, cannot pad
        "611981640001600",  # 15 digits
        "6119816400016X",  # non-numeric
        "611981640001",  # 12 digits -- parse only pads 13 -> 14
    ],
)
def test_parse_rejects_a_value_it_cannot_normalise(raw: str) -> None:
    with pytest.raises(InvalidCnpjError):
        Cnpj.parse(raw)


@pytest.mark.unit
def test_constructor_rejects_valid_length_wrong_check_digits() -> None:
    with pytest.raises(InvalidCnpjError):
        Cnpj("61198164000161")


@pytest.mark.unit
@pytest.mark.parametrize("value", _MANIFEST_CNPJS)
def test_every_manifest_cnpj_has_correct_check_digits(value: str) -> None:
    assert _cnpj_check_digits(value[:12]) == value[12:]
    assert Cnpj(value).value == value


@pytest.mark.unit
def test_is_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        Cnpj(_VALID).value = "x"  # type: ignore[misc]
