"""The pinned exclusion co-retrieval contract and its anti-drift guards -- [M3-06]."""

import re

import pytest

from infrastructure.rag import exclusion_co_retrieval_config as config
from infrastructure.rag.exclusion_co_retrieval_config import (
    ADJACENT_SECTION_MAX_PAGE_GAP,
    CROSS_REFERENCE_PATTERN,
    RESERVED_EXCLUSION_SLOTS,
    config_fingerprint,
)


@pytest.mark.unit
def test_contract_constants() -> None:
    assert RESERVED_EXCLUSION_SLOTS >= 1
    assert ADJACENT_SECTION_MAX_PAGE_GAP >= 0
    # The pattern must compile and capture the numbering token.
    match = re.search(CROSS_REFERENCE_PATTERN, "conforme cláusula 10.2", re.IGNORECASE)
    assert match is not None and match.group(1) == "10.2"


@pytest.mark.unit
def test_config_fingerprint_is_stable() -> None:
    # A pinned literal, so reordering or dropping a field from the payload is a
    # visible test failure.
    assert config_fingerprint() == "7ed4c97c4e8f1cb4"
    assert re.fullmatch(r"[0-9a-f]{16}", config_fingerprint()) is not None


@pytest.mark.unit
@pytest.mark.parametrize(
    ("attr", "value"),
    [
        ("RESERVED_EXCLUSION_SLOTS", 99),
        ("ADJACENT_SECTION_MAX_PAGE_GAP", 99),
        ("CROSS_REFERENCE_PATTERN", r"artigos?\s+(\d+)"),
    ],
)
def test_every_contract_field_changes_the_fingerprint(
    monkeypatch: pytest.MonkeyPatch, attr: str, value: object
) -> None:
    baseline = config_fingerprint()
    monkeypatch.setattr(config, attr, value)
    assert config_fingerprint() != baseline
