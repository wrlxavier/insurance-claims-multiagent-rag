"""Tests for the synthetic-claim PII safety net [M2-04].

Table-driven: known-bad strings must flag, and clean sentences containing
this corpus's acronyms/insurer names must not trip the plate/CPF patterns
specifically (the proper-name heuristic is allowed documented false
positives -- see the module docstring).
"""

import pytest
from scripts.pii_safety_net import scan_narrative_for_pii

BAD_CASES = [
    ("plate old format with dash", "o carro placa ABC-1234 bateu", "plate_old_format"),
    ("plate old format no dash", "o carro placa ABC1234 bateu", "plate_old_format"),
    ("plate mercosul format", "a placa era ABC1D23", "plate_mercosul_format"),
    ("cpf with punctuation", "meu cpf é 123.456.789-00", "cpf_like"),
    ("cpf without punctuation", "meu cpf é 12345678900", "cpf_like"),
]


@pytest.mark.unit
@pytest.mark.parametrize(("case_id", "text", "expected_label"), BAD_CASES)
def test_scan_narrative_for_pii_flags_known_bad_patterns(
    case_id: str, text: str, expected_label: str
) -> None:
    hits = scan_narrative_for_pii(text)
    assert expected_label in hits, case_id


CLEAN_CASES = [
    ("plain narrative", "meu carro bateu na traseira de outro veículo ontem"),
    ("insurer acronym", "sou segurado da SUSEP e da RCF-A, tudo certo"),
    ("casco acronym", "o sinistro é sobre o seguro CASCO do meu carro"),
]


@pytest.mark.unit
@pytest.mark.parametrize(("case_id", "text"), CLEAN_CASES)
def test_scan_narrative_for_pii_does_not_flag_plate_or_cpf_on_clean_text(
    case_id: str, text: str
) -> None:
    hits = scan_narrative_for_pii(text)
    assert "plate_old_format" not in hits, case_id
    assert "plate_mercosul_format" not in hits, case_id
    assert "cpf_like" not in hits, case_id


@pytest.mark.unit
def test_scan_narrative_for_pii_returns_empty_list_for_clean_text() -> None:
    assert scan_narrative_for_pii("meu carro bateu ontem, foi tranquilo") == []
