"""Tests for the corpus-artifacts packaging script."""

from pathlib import Path

import pytest
from scripts.fetch_corpus_artifacts import (
    extract_archive,
    read_expected_checksum,
    verify_checksum,
)
from scripts.package_corpus_artifacts import build_archive, write_checksum


@pytest.mark.unit
def test_build_archive_raises_on_missing_source_path(tmp_path: Path) -> None:
    missing = tmp_path / "build"

    with pytest.raises(FileNotFoundError, match="Missing source path"):
        build_archive((missing,), tmp_path / "archive.tar.gz")


@pytest.mark.unit
def test_build_archive_and_checksum_round_trip_through_fetch_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """What package_corpus_artifacts packs, fetch_corpus_artifacts must unpack.

    ``build_archive`` uses each source path's own string form as the tar
    arcname, so -- matching real usage, where the script always runs from
    the repo root with relative ``SOURCE_PATHS`` -- this test runs from a
    working directory of ``tmp_path`` with relative paths, not absolute ones.
    """
    monkeypatch.chdir(tmp_path)
    build_dir = Path("build")
    build_dir.mkdir()
    (build_dir / "manifest.json").write_text(
        '{"total_clause_count": 4925}', encoding="utf-8"
    )
    cache_file = Path("cache.jsonl")
    cache_file.write_text('{"key": "value"}\n', encoding="utf-8")

    archive_path = Path("dist") / "corpus-artifacts.tar.gz"
    checksum_path = Path("dist") / "corpus-artifacts.sha256"
    build_archive((build_dir, cache_file), archive_path)
    write_checksum(archive_path, checksum_path)

    expected_sha256 = read_expected_checksum(checksum_path)
    verify_checksum(archive_path, expected_sha256)  # does not raise

    dest_dir = Path("dest")
    members = extract_archive(archive_path, dest_dir)

    assert (dest_dir / "build" / "manifest.json").read_text(
        encoding="utf-8"
    ) == '{"total_clause_count": 4925}'
    assert (dest_dir / "cache.jsonl").read_text(
        encoding="utf-8"
    ) == '{"key": "value"}\n'
    assert "build/manifest.json" in members
