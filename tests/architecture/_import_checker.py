"""Reusable static import checker for Clean Architecture boundaries.

Parses Python source with ``ast`` — it never executes the modules it
inspects — and reports any import whose top-level module name matches a
forbidden root. Kept generic on purpose so the same helper can be reused
for the application-layer check in M5-02 without being rewritten.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ForbiddenImport:
    file: Path
    line: int
    module: str


def _top_level_module(name: str) -> str:
    """Return the root package of a dotted module path, e.g. 'a.b.c' -> 'a'."""
    return name.split(".", 1)[0]


def _iter_imported_modules(tree: ast.Module) -> list[tuple[int, str]]:
    """Collect (line_number, module_name) for every import in the tree."""
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.append((node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom):
            # Relative imports (from . import x) have module=None; they can
            # only refer to something inside the same package, so they are
            # never a boundary violation and can be skipped.
            if node.module is not None:
                found.append((node.lineno, node.module))
    return found


def find_forbidden_imports(
    package_dir: Path,
    forbidden_roots: frozenset[str],
) -> list[ForbiddenImport]:
    """Scan every .py file under package_dir for forbidden top-level imports.

    Args:
        package_dir: Root directory of the layer being checked
            (e.g. app/src/domain).
        forbidden_roots: Set of top-level module names that must not be
            imported from within package_dir (e.g. {"fastapi", "sqlalchemy"}).

    Returns:
        One ForbiddenImport per violation found, empty if the layer is clean.
    """
    violations: list[ForbiddenImport] = []

    for py_file in sorted(package_dir.rglob("*.py")):
        source = py_file.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(py_file))

        for line, module_name in _iter_imported_modules(tree):
            root = _top_level_module(module_name)
            if root in forbidden_roots:
                violations.append(
                    ForbiddenImport(file=py_file, line=line, module=module_name)
                )

    return violations


def format_violations(violations: list[ForbiddenImport]) -> str:
    """Render violations as a readable multi-line report for assert messages."""
    lines = [
        f"  {v.file}:{v.line} imports '{v.module}'"
        for v in violations
    ]
    return "\n".join(lines)
