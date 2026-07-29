"""Guard the M0-06 verdict vocabulary: compatible / incompatible /
insufficient_information, never covered / denied.

Statically scans every prompt template and the project's enum module for
forbidden vocabulary, so a prompt or schema rewritten later cannot
silently drift back to a covered/denied framing. See docs/SCOPE.md for the
canonical statement this guards.
"""

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

PROMPTS_DIR = REPO_ROOT / "app" / "src" / "infrastructure" / "graph" / "prompts"
ENUMS_FILE = REPO_ROOT / "app" / "src" / "infrastructure" / "config" / "enums.py"

FORBIDDEN_PATTERN = re.compile(
    r"\b(covered|uncovered|denied|denial|coverage decision)\b", re.IGNORECASE
)


def _scan_file(path: Path) -> list[str]:
    violations = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for line_no, line in enumerate(lines, start=1):
        match = FORBIDDEN_PATTERN.search(line)
        if match:
            violations.append(f"  {path}:{line_no} contains '{match.group(0)}'")
    return violations


@pytest.mark.unit
def test_prompts_directory_exists() -> None:
    assert PROMPTS_DIR.is_dir(), f"expected shared prompts package at {PROMPTS_DIR}"
    assert list(PROMPTS_DIR.glob("*.py")), f"expected a prompt module in {PROMPTS_DIR}"


@pytest.mark.unit
def test_no_forbidden_verdict_vocabulary_in_prompts() -> None:
    violations = []
    for py_file in sorted(PROMPTS_DIR.rglob("*.py")):
        violations.extend(_scan_file(py_file))
    report = "\n".join(violations)
    assert not violations, f"forbidden verdict vocabulary found in prompts:\n{report}"


@pytest.mark.unit
def test_no_forbidden_verdict_vocabulary_in_enums() -> None:
    violations = _scan_file(ENUMS_FILE) if ENUMS_FILE.exists() else []
    assert not violations, "forbidden verdict vocabulary found in enums:\n" + "\n".join(
        violations
    )
