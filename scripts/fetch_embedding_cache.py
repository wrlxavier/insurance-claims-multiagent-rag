#!/usr/bin/env python3
"""Fetch the pre-computed embedding cache from GitHub Releases -- [M5-09].

`make embed-chunks` costs real wall-clock time even though it has no dollar
price: embedding the full 4,540-chunk corpus cold takes ~41 minutes of CPU
(`docs/EMBEDDINGS.md`'s "Corpus embedding cost"), because the pinned model
(`Alibaba-NLP/gte-multilingual-base`) runs in-process, not behind an API.
Anyone who only wants to try the assessment flow should not have to pay that
wait.

This is the exact sibling of `fetch_corpus_artifacts.py`, one stage later in
the pipeline: that script skips the LLM-cost parsing stage; this one skips
the compute-cost embedding stage. It downloads a single release asset -- a
tarball of `data/cache/embeddings/cache.jsonl`, the content-addressed
embedding cache `infrastructure.rag.embedding_cache.CachingEmbedder` reads
and writes -- verifies its checksum, and extracts it on top of the existing
(gitignored) directory structure. It never touches anything already
committed to git, and it needs no `.env`/LLM credentials (embedding is
local, not an LLM call) -- this is a plain download.

With the cache in place, `make build-index`'s `embed-chunks` step becomes a
cache-hit replay (~2.6s, measured in `docs/EMBEDDINGS.md`) instead of a cold
41-minute run: every chunk's `sha256(fingerprint · text)` key is already
present, so `embed_missing_chunks` never loads the model.

Run via `make fetch-embedding-cache`, or `make fetch-demo-artifacts` for both
this and `fetch_corpus_artifacts.py` in one step. See
`scripts/package_embedding_cache.py` for the maintainer-side counterpart that
produces the release asset.
"""

from __future__ import annotations

import hashlib
import tarfile
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

GITHUB_REPO = "wrlxavier/insurance-claims-multiagent-rag"

# Pinned, not "latest" -- see fetch_corpus_artifacts.py's identical rationale.
# Bump this when publishing a new version via `make package-embedding-cache`
# plus a new GitHub Release.
ARTIFACTS_RELEASE_TAG = "m3-embedding-cache-v1"

ARCHIVE_ASSET_NAME = "embedding-cache.tar.gz"
CHECKSUM_ASSET_NAME = "embedding-cache.sha256"

RELEASE_BASE_URL = (
    f"https://github.com/{GITHUB_REPO}/releases/download/{ARTIFACTS_RELEASE_TAG}"
)

DOWNLOAD_TIMEOUT_SECONDS = 60
EXTRACT_ROOT = Path(".")


def download_file(url: str, dest: Path) -> None:
    """Download `url` to `dest`, creating parent directories as needed."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(  # noqa: S310 -- fixed https:// GitHub Releases URL
            url, timeout=DOWNLOAD_TIMEOUT_SECONDS
        ) as response:
            dest.write_bytes(response.read())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            f"Failed to download {url} ({exc.code} {exc.reason}). "
            f"Has release `{ARTIFACTS_RELEASE_TAG}` been published on "
            f"github.com/{GITHUB_REPO}/releases?"
        ) from exc


def read_expected_checksum(checksum_path: Path) -> str:
    """Parse a `sha256sum`-format file (`<hex digest>  <filename>`)."""
    content = checksum_path.read_text(encoding="utf-8").strip()
    digest, _, _name = content.partition(" ")
    if len(digest) != 64:
        raise ValueError(f"Malformed checksum file {checksum_path}: {content!r}")
    return digest


def verify_checksum(archive_path: Path, expected_sha256: str) -> None:
    """Raise ValueError if `archive_path`'s SHA-256 does not match `expected_sha256`."""
    actual = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    if actual != expected_sha256:
        raise ValueError(
            f"Checksum mismatch for {archive_path}: expected {expected_sha256}, "
            f"got {actual}. The download may be corrupted or incomplete -- try again."
        )


def extract_archive(archive_path: Path, dest_dir: Path) -> list[str]:
    """Extract `archive_path` into `dest_dir`. Returns the extracted member names."""
    with tarfile.open(archive_path, "r:gz") as archive:
        archive.extractall(dest_dir, filter="data")
        return archive.getnames()


def main() -> None:
    """Download, verify and extract the pinned embedding-cache release."""
    archive_url = f"{RELEASE_BASE_URL}/{ARCHIVE_ASSET_NAME}"
    checksum_url = f"{RELEASE_BASE_URL}/{CHECKSUM_ASSET_NAME}"

    with tempfile.TemporaryDirectory(prefix="embedding-cache-") as tmp:
        tmp_dir = Path(tmp)
        archive_path = tmp_dir / ARCHIVE_ASSET_NAME
        checksum_path = tmp_dir / CHECKSUM_ASSET_NAME

        print(f"Downloading {checksum_url} ...")
        download_file(checksum_url, checksum_path)
        expected_sha256 = read_expected_checksum(checksum_path)

        print(f"Downloading {archive_url} ...")
        download_file(archive_url, archive_path)

        print("Verifying checksum ...")
        verify_checksum(archive_path, expected_sha256)

        print(f"Extracting into {EXTRACT_ROOT.resolve()} ...")
        members = extract_archive(archive_path, EXTRACT_ROOT)

    print(f"Fetched {ARTIFACTS_RELEASE_TAG}: wrote {len(members)} files.")


if __name__ == "__main__":
    main()
