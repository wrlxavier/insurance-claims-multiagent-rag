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
GRAPH_DIR = REPO_ROOT / "app" / "src" / "infrastructure" / "graph"
ENUMS_FILE = REPO_ROOT / "app" / "src" / "infrastructure" / "config" / "enums.py"
DOMAIN_DIR = REPO_ROOT / "app" / "src" / "domain"

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
def test_no_forbidden_verdict_vocabulary_in_graph() -> None:
    # The whole graph package, not just prompts/: [M4-01]'s state schema
    # carries the verdict enum reference and the citation/audit types, and a
    # covered/denied literal must not drift into any of them either.
    violations = []
    for py_file in sorted(GRAPH_DIR.rglob("*.py")):
        violations.extend(_scan_file(py_file))
    report = "\n".join(violations)
    assert not violations, f"forbidden verdict vocabulary found in graph:\n{report}"


@pytest.mark.unit
def test_no_forbidden_verdict_vocabulary_in_domain_and_enums() -> None:
    # The whole domain package, not just verdict.py: [M5-01]'s entities carry
    # the verdict enum reference, the citation types and free-text
    # ``reasoning`` / ``recommended_action`` semantics on ``Assessment`` -- a
    # covered/denied literal must not drift into any of them either.
    violations = _scan_file(ENUMS_FILE) if ENUMS_FILE.exists() else []
    for py_file in sorted(DOMAIN_DIR.rglob("*.py")):
        violations.extend(_scan_file(py_file))
    assert not violations, "forbidden verdict vocabulary found in domain/enums:\n" + (
        "\n".join(violations)
    )
