import pytest

from infrastructure.parsing.boilerplate_caching import (
    boilerplate_cache_path,
    compute_boilerplate_cache_key,
)


@pytest.mark.unit
def test_compute_boilerplate_cache_key_is_deterministic() -> None:
    assert compute_boilerplate_cache_key("v1") == compute_boilerplate_cache_key("v1")


@pytest.mark.unit
def test_compute_boilerplate_cache_key_changes_with_version() -> None:
    assert compute_boilerplate_cache_key("v1") != compute_boilerplate_cache_key("v2")


@pytest.mark.unit
def test_boilerplate_cache_path_names_file_by_document_id_and_key() -> None:
    path = boilerplate_cache_path("10", "abc123")

    assert path.name == "10__abc123.parquet"
