#!/usr/bin/env python3
"""Fetch the pre-computed M1 corpus artifacts from GitHub Releases.

Running the full parsing pipeline (OCR, LLM clause classification, the
vision-LLM boundary-escalation pass, LLM validation) costs real tokens and
real wall-clock time -- on the order of hours for the escalation pass alone
(see ``docs/PARSING.md``'s "Known limitations"). Anyone who only wants to
inspect the already-published result should not have to pay that cost.

This script downloads a single release asset -- a tarball of the four
LLM-cost-bearing artifacts (the finished corpus under ``build/``, the two
LLM classification/escalation caches, and the validation judgment JSONs
under ``eval/temp/sample_validations/``) -- verifies its checksum, and
extracts it on top of the existing (gitignored) directory structure. It
never touches ``data/policies/raw/`` or anything already committed to git,
and it needs no ``.env``/LLM credentials -- this is a plain download.

Large-but-cheap-to-regenerate caches (rasterized page images, ~312MB
combined) are intentionally not part of this bundle: they carry no LLM cost
and reconstruct deterministically from the already-committed PDFs, so
shipping them would just be dead weight. Run `make parse` /
`make escalate-vision-boundaries` / `make validate-parsing-quality-sample`
if you need those instead.

Run via `make fetch-corpus-artifacts`. See `scripts/package_corpus_artifacts.py`
for the maintainer-side counterpart that produces the release assets.
"""

from __future__ import annotations

import hashlib
import tarfile
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

GITHUB_REPO = "wrlxavier/insurance-claims-multiagent-rag"

# Pinned, not "latest" -- a given commit should always resolve to the exact
# artifacts it was built and verified against. Bump this when publishing a
# new version via `make package-corpus-artifacts` plus a new GitHub Release,
# per that script's docstring.
ARTIFACTS_RELEASE_TAG = "m1-artifacts-v1"

ARCHIVE_ASSET_NAME = "corpus-artifacts.tar.gz"
CHECKSUM_ASSET_NAME = "corpus-artifacts.sha256"

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
    """Download, verify and extract the pinned corpus-artifacts release."""
    archive_url = f"{RELEASE_BASE_URL}/{ARCHIVE_ASSET_NAME}"
    checksum_url = f"{RELEASE_BASE_URL}/{CHECKSUM_ASSET_NAME}"

    with tempfile.TemporaryDirectory(prefix="corpus-artifacts-") as tmp:
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
