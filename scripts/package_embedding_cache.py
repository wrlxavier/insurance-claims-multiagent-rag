#!/usr/bin/env python3
"""Package the embedding cache for a GitHub Release -- [M5-09].

Maintainer-only. The exact sibling of `package_corpus_artifacts.py`: bundles
`data/cache/embeddings/cache.jsonl` (produced by `make embed-chunks` against
a fully-loaded `chunk` table -- see `docs/EMBEDDINGS.md`) into a single
tarball plus a SHA-256 checksum file under `dist/` (already gitignored),
ready to attach to a GitHub Release.

After running this:

1. Publish both files, e.g. `gh release create <tag>
   dist/embedding-cache.tar.gz dist/embedding-cache.sha256 --title "<title>"
   --notes "<notes>"` (or the GitHub web UI).
2. Bump `ARTIFACTS_RELEASE_TAG` in `scripts/fetch_embedding_cache.py` to
   `<tag>` and commit.

Run via `make package-embedding-cache`.
"""

from __future__ import annotations

import hashlib
import tarfile
from pathlib import Path

OUTPUT_DIR = Path("dist")
ARCHIVE_PATH = OUTPUT_DIR / "embedding-cache.tar.gz"
CHECKSUM_PATH = OUTPUT_DIR / "embedding-cache.sha256"

# Kept in sync with fetch_embedding_cache.py's module docstring -- what one
# script packs, the other must unpack.
SOURCE_PATHS = (Path("data/cache/embeddings/cache.jsonl"),)


def build_archive(source_paths: tuple[Path, ...], archive_path: Path) -> None:
    """Tar+gzip every path in `source_paths`, preserving their relative layout."""
    missing = [path for path in source_paths if not path.exists()]
    if missing:
        joined_missing = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            f"Missing source path(s): {joined_missing}. Run `make build-index` "
            "(or at least `make embed-chunks` against a fully-loaded chunk "
            "table) first."
        )
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "w:gz") as archive:
        for path in source_paths:
            archive.add(path, arcname=str(path))


def write_checksum(archive_path: Path, checksum_path: Path) -> None:
    """Write a `sha256sum`-format checksum file for `archive_path`."""
    digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    checksum_path.write_text(f"{digest}  {archive_path.name}\n", encoding="utf-8")


def main() -> None:
    """Build the release tarball and its checksum under dist/."""
    build_archive(SOURCE_PATHS, ARCHIVE_PATH)
    write_checksum(ARCHIVE_PATH, CHECKSUM_PATH)
    size_mb = ARCHIVE_PATH.stat().st_size / 1_000_000
    print(f"Wrote {ARCHIVE_PATH} ({size_mb:.1f}MB) and {CHECKSUM_PATH}.")
    print(
        "Publish both to a GitHub Release, e.g.:\n"
        f"  gh release create <tag> {ARCHIVE_PATH} {CHECKSUM_PATH} "
        '--title "<title>" --notes "<notes>"\n'
        "then bump ARTIFACTS_RELEASE_TAG in scripts/fetch_embedding_cache.py."
    )


if __name__ == "__main__":
    main()
