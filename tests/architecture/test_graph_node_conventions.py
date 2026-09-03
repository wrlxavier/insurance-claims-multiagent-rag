"""Enforce the [M4-01b] node-authoring conventions.

A graph node is a module-level function ``(state, runtime) -> dict`` -- no
class-based nodes, and the first two parameters are named ``state`` and
``runtime`` (the name LangGraph's runtime injector binds by). Statically scans
every module under ``infrastructure/graph/nodes/`` with ``ast`` -- it never
imports them -- so the five M4 node issues cannot converge on five different
node shapes. See ``docs/ARCHITECTURE.md`` for the convention this guards.
"""

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GRAPH_DIR = REPO_ROOT / "app" / "src" / "infrastructure" / "graph"
NODES_DIR = GRAPH_DIR / "nodes"

_FUNCTION_DEFS = (ast.FunctionDef, ast.AsyncFunctionDef)


def _node_modules() -> list[Path]:
    return sorted(p for p in NODES_DIR.glob("*.py") if p.name != "__init__.py")


def _base_names(class_def: ast.ClassDef) -> list[str]:
    names: list[str] = []
    for base in class_def.bases:
        if isinstance(base, ast.Name):
            names.append(base.id)
        elif isinstance(base, ast.Attribute):
            names.append(base.attr)
    return names


def _class_based_node_violations(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        method_names = {
            child.name for child in node.body if isinstance(child, _FUNCTION_DEFS)
        }
        subclasses_runnable = any("Runnable" in name for name in _base_names(node))
        if "__call__" in method_names or subclasses_runnable:
            violations.append(
                f"  {path}:{node.lineno} defines class-based node '{node.name}'"
            )
    return violations


def _signature_violations(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    violations: list[str] = []
    for node in tree.body:
        if not isinstance(node, _FUNCTION_DEFS) or node.name.startswith("_"):
            continue
        params = [arg.arg for arg in node.args.posonlyargs + node.args.args]
        location = f"  {path}:{node.lineno} '{node.name}'"
        if params[:1] != ["state"]:
            violations.append(f"{location} first parameter is not 'state'")
        elif len(params) >= 2 and params[1] != "runtime":
            violations.append(f"{location} second parameter is not 'runtime'")
    return violations


@pytest.mark.unit
def test_graph_scaffolding_exists() -> None:
    # Guards against the scans below passing only because a path is missing.
    assert NODES_DIR.is_dir(), f"expected the nodes package at {NODES_DIR}"
    for name in ("schemas.py", "context.py"):
        assert (GRAPH_DIR / name).is_file(), f"expected infrastructure/graph/{name}"


@pytest.mark.unit
def test_no_class_based_nodes() -> None:
    violations: list[str] = []
    for module in _node_modules():
        violations.extend(_class_based_node_violations(module))
    assert not violations, "class-based graph nodes are forbidden:\n" + "\n".join(
        violations
    )


@pytest.mark.unit
def test_exported_node_functions_take_state_and_runtime() -> None:
    violations: list[str] = []
    for module in _node_modules():
        violations.extend(_signature_violations(module))
    assert not violations, "graph node signature convention violated:\n" + "\n".join(
        violations
    )
