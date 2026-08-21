"""Tests for the corpus-artifacts fetch script."""

import hashlib
import tarfile
import urllib.request
from pathlib import Path

import pytest
from scripts.fetch_corpus_artifacts import (
    download_file,
    extract_archive,
    read_expected_checksum,
    verify_checksum,
)


@pytest.mark.unit
def test_read_expected_checksum_parses_sha256sum_format(tmp_path: Path) -> None:
    checksum_path = tmp_path / "corpus-artifacts.sha256"
    digest = "a" * 64
    checksum_path.write_text(f"{digest}  corpus-artifacts.tar.gz\n", encoding="utf-8")

    assert read_expected_checksum(checksum_path) == digest


@pytest.mark.unit
def test_read_expected_checksum_rejects_malformed_content(tmp_path: Path) -> None:
    checksum_path = tmp_path / "corpus-artifacts.sha256"
    checksum_path.write_text("not-a-valid-digest\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Malformed checksum file"):
        read_expected_checksum(checksum_path)


@pytest.mark.unit
def test_verify_checksum_passes_for_matching_digest(tmp_path: Path) -> None:
    archive_path = tmp_path / "archive.tar.gz"
    archive_path.write_bytes(b"some archive bytes")
    expected = hashlib.sha256(archive_path.read_bytes()).hexdigest()

    verify_checksum(archive_path, expected)


@pytest.mark.unit
def test_verify_checksum_raises_on_mismatch(tmp_path: Path) -> None:
    archive_path = tmp_path / "archive.tar.gz"
    archive_path.write_bytes(b"some archive bytes")

    with pytest.raises(ValueError, match="Checksum mismatch"):
        verify_checksum(archive_path, "0" * 64)


@pytest.mark.unit
def test_extract_archive_writes_files_preserving_relative_layout(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "source"
    (source_dir / "build").mkdir(parents=True)
    (source_dir / "build" / "manifest.json").write_text("{}", encoding="utf-8")
    cache_dir = source_dir / "data" / "cache" / "llm_classification"
    cache_dir.mkdir(parents=True)
    (cache_dir / "cache.jsonl").write_text('{"key": "value"}\n', encoding="utf-8")

    archive_path = tmp_path / "archive.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(source_dir / "build", arcname="build")
        archive.add(
            cache_dir / "cache.jsonl",
            arcname="data/cache/llm_classification/cache.jsonl",
        )

    dest_dir = tmp_path / "dest"
    members = extract_archive(archive_path, dest_dir)

    assert (dest_dir / "build" / "manifest.json").read_text(encoding="utf-8") == "{}"
    assert (
        dest_dir / "data" / "cache" / "llm_classification" / "cache.jsonl"
    ).read_text(encoding="utf-8") == '{"key": "value"}\n'
    assert "build/manifest.json" in members
    assert "data/cache/llm_classification/cache.jsonl" in members


@pytest.mark.unit
def test_download_file_writes_response_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _FakeResponse:
        def __enter__(self) -> "_FakeResponse":
            return self

        def __exit__(self, *exc_info: object) -> None:
            return None

        def read(self) -> bytes:
            return b"fake bytes"

    def _fake_urlopen(url: str, timeout: float | None = None) -> _FakeResponse:
        assert url == "https://example.com/file"
        return _FakeResponse()

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)

    dest = tmp_path / "nested" / "file.bin"
    download_file("https://example.com/file", dest)

    assert dest.read_bytes() == b"fake bytes"
