"""The pinned lexical-retrieval contract and its anti-drift guards -- [M3-03]."""

import re

import pytest

from infrastructure.rag import lexical_config
from infrastructure.rag.lexical_config import (
    BM25_B,
    BM25_K1,
    IDF_VARIANT,
    LEXICAL_ANALYZER_VERSION,
    LEXICAL_INDEX_TEXT_FIELD,
    config_fingerprint,
)

_SEED = frozenset({"casco", "rcf", "vmr", "vd"})


@pytest.mark.unit
def test_contract_constants() -> None:
    assert LEXICAL_ANALYZER_VERSION == "v1"
    assert BM25_K1 == 1.5
    assert BM25_B == 0.75
    assert IDF_VARIANT == "lucene_plus_one"
    assert LEXICAL_INDEX_TEXT_FIELD == "text"


@pytest.mark.unit
def test_config_fingerprint_is_stable() -> None:
    # A pinned literal: reordering or dropping a field from the payload, or
    # changing a constant without re-running the eval, is a visible failure.
    assert config_fingerprint(exception_tokens=_SEED) == "d3c1c336bda424e0"
    assert re.fullmatch(r"[0-9a-f]{16}", config_fingerprint(exception_tokens=_SEED))


@pytest.mark.unit
def test_a_different_exception_set_moves_the_fingerprint() -> None:
    baseline = config_fingerprint(exception_tokens=_SEED)
    assert config_fingerprint(exception_tokens=_SEED | {"franquia"}) != baseline
    assert config_fingerprint(exception_tokens=frozenset()) != baseline


@pytest.mark.unit
@pytest.mark.parametrize(
    ("attr", "value"),
    [
        ("LEXICAL_ANALYZER_VERSION", "v2"),
        ("STEMMER_LANGUAGE", "spanish"),
        ("BM25_K1", 1.2),
        ("BM25_B", 0.4),
        ("IDF_VARIANT", "okapi"),
        ("LEXICAL_INDEX_TEXT_FIELD", "display_text"),
    ],
)
def test_every_contract_field_changes_the_fingerprint(
    monkeypatch: pytest.MonkeyPatch, attr: str, value: object
) -> None:
    baseline = config_fingerprint(exception_tokens=_SEED)
    monkeypatch.setattr(lexical_config, attr, value)
    assert config_fingerprint(exception_tokens=_SEED) != baseline
