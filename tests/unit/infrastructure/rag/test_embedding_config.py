"""The pinned embedding contract and its anti-drift guards -- [M3-02]."""

import re
from pathlib import Path

import pytest

from infrastructure.rag.embedding_config import (
    DISTANCE_METRIC,
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MAX_INPUT_TOKENS,
    EMBEDDING_MODEL_ID,
    EMBEDDING_MODEL_REVISION,
    EMBEDDING_TRUST_REMOTE_CODE,
    NORMALIZE_EMBEDDINGS,
    PASSAGE_PREFIX,
    QUERY_PREFIX,
    DistanceMetric,
    format_passage,
    format_query,
)

_REPO_ROOT = Path(__file__).resolve().parents[4]
_APP_SRC = _REPO_ROOT / "app" / "src"
_CONFIG_MODULE = _APP_SRC / "infrastructure" / "rag" / "embedding_config.py"

# The query/passage prefix each candidate model requires. gte-multilingual-base
# takes none; the E5 family's model card mandates "query: " / "passage: ".
# Swapping EMBEDDING_MODEL_ID without reconciling QUERY_PREFIX / PASSAGE_PREFIX
# fails test_prefixes_match_the_pinned_model below.
_KNOWN_MODEL_PREFIXES: dict[str, tuple[str, str]] = {
    "Alibaba-NLP/gte-multilingual-base": ("", ""),
    "BAAI/bge-m3": ("", ""),
    "intfloat/multilingual-e5-large": ("query: ", "passage: "),
    "intfloat/multilingual-e5-base": ("query: ", "passage: "),
}


@pytest.mark.unit
def test_model_contract_constants() -> None:
    assert EMBEDDING_MODEL_ID == "Alibaba-NLP/gte-multilingual-base"
    assert EMBEDDING_DIMENSIONS == 768
    assert EMBEDDING_MAX_INPUT_TOKENS == 8192
    assert DISTANCE_METRIC is DistanceMetric.COSINE
    assert NORMALIZE_EMBEDDINGS is True
    assert EMBEDDING_TRUST_REMOTE_CODE is True


@pytest.mark.unit
def test_revision_is_a_pinned_commit_not_a_floating_alias() -> None:
    # A 40-char hex SHA -- never "main"/"master"/"" -- so a provider-side
    # re-upload cannot change the vectors behind a published number.
    assert re.fullmatch(r"[0-9a-f]{40}", EMBEDDING_MODEL_REVISION) is not None


@pytest.mark.unit
def test_format_functions_apply_the_configured_prefixes() -> None:
    # The single formatting path both the index side and [M3-04]'s query side
    # must use. Identity for the current (prefix-free) model.
    assert format_query("Cláusula 3.1") == f"{QUERY_PREFIX}Cláusula 3.1"
    assert format_passage("Cláusula 3.1") == f"{PASSAGE_PREFIX}Cláusula 3.1"


@pytest.mark.unit
def test_prefixes_match_the_pinned_model() -> None:
    assert EMBEDDING_MODEL_ID in _KNOWN_MODEL_PREFIXES, (
        f"{EMBEDDING_MODEL_ID} is new here: record its query/passage prefix "
        "requirement in _KNOWN_MODEL_PREFIXES and in embedding_config, then "
        "confirm both format_* call sites were updated together."
    )
    assert _KNOWN_MODEL_PREFIXES[EMBEDDING_MODEL_ID] == (QUERY_PREFIX, PASSAGE_PREFIX)


def _app_src_modules_except_config() -> list[Path]:
    return [
        path
        for path in _APP_SRC.rglob("*.py")
        if path.resolve() != _CONFIG_MODULE.resolve()
    ]


@pytest.mark.unit
def test_prefix_constants_are_defined_in_exactly_one_module() -> None:
    assignment = re.compile(r"^\s*(QUERY_PREFIX|PASSAGE_PREFIX)\s*=", re.MULTILINE)
    offenders = [
        str(path.relative_to(_REPO_ROOT))
        for path in _app_src_modules_except_config()
        if assignment.search(path.read_text(encoding="utf-8"))
    ]
    assert not offenders, (
        f"QUERY_PREFIX / PASSAGE_PREFIX assigned outside embedding_config.py: "
        f"{offenders}. Import format_query / format_passage instead."
    )


@pytest.mark.unit
def test_no_instruction_prefix_literal_is_hardcoded_anywhere() -> None:
    # The E5-style instruction prefixes must only ever exist as
    # embedding_config's QUERY_PREFIX / PASSAGE_PREFIX, never inline on one
    # side of the pipeline -- that is exactly the index-vs-query drift the DoD
    # asks to make impossible.
    literals = ('"query: "', "'query: '", '"passage: "', "'passage: '")
    offenders = [
        str(path.relative_to(_REPO_ROOT))
        for path in _app_src_modules_except_config()
        if any(literal in path.read_text(encoding="utf-8") for literal in literals)
    ]
    assert not offenders, f"hardcoded instruction prefix in: {offenders}"


@pytest.mark.unit
def test_env_example_embedding_model_matches_the_pinned_id() -> None:
    env_example = (_REPO_ROOT / ".env.example").read_text(encoding="utf-8")
    values = {
        key.strip(): value.strip().strip('"').strip("'")
        for line in env_example.splitlines()
        if line.strip() and not line.lstrip().startswith("#") and "=" in line
        for key, _, value in [line.partition("=")]
    }
    assert values["EMBEDDING_MODEL"] == EMBEDDING_MODEL_ID
