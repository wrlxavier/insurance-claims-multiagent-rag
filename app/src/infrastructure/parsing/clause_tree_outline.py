"""Human-reviewable indented-outline rendering of a clause tree.

Distinct from [infrastructure.parsing.clause_tree_report]'s aggregate
Markdown table: this renders one document's full structure so a reviewer
can sanity-check the heading heuristic before a corpus-wide run, per
[M1-04]'s checkpoint requirement. Content lines are deliberately omitted --
only titles and structure -- to keep it scannable.
"""

from domain.clause_tree import Clause, ClauseTree

_TITLE_MAX_LENGTH = 100


def _truncate(title: str) -> str:
    if len(title) <= _TITLE_MAX_LENGTH:
        return title
    return title[: _TITLE_MAX_LENGTH - 1] + "…"


def _render_clause(clause: Clause, by_id: dict[str, Clause], indent: int) -> list[str]:
    label = clause.numbering_label or "part"
    pages = (
        f"p.{clause.page_start}"
        if clause.page_start == clause.page_end
        else f"p.{clause.page_start}-{clause.page_end}"
    )
    anomaly = " ⚠depth" if clause.is_depth_anomaly else ""
    prefix = "  " * indent + "- "
    rendered = [f"{prefix}[{label}] {_truncate(clause.title)}  ({pages}){anomaly}"]
    for child_id in clause.child_ids:
        rendered.extend(_render_clause(by_id[child_id], by_id, indent + 1))
    return rendered


def render_outline(tree: ClauseTree) -> str:
    """Render a full clause tree as an indented outline for human review."""
    by_id = {clause.clause_id: clause for clause in tree.all_clauses}
    lines = [
        f"# Clause tree checkpoint: {tree.filename} (document {tree.document_id})",
        "",
        f"Extraction mode: {tree.report.extraction_mode} | "
        f"Clauses: {tree.report.clause_count} | "
        f"Max depth: {tree.report.max_depth} | "
        f"Orphan ratio: {tree.report.orphan_ratio:.3f} "
        f"({tree.report.orphan_char_count}/{tree.report.total_char_count} chars) | "
        f"Warnings: {len(tree.report.warnings)}",
        "",
    ]
    for root in tree.roots:
        lines.extend(_render_clause(root, by_id, indent=0))

    if tree.report.warnings:
        lines.append("")
        for warning in tree.report.warnings:
            lines.append(
                f"⚠ {warning.kind} at p.{warning.page_number}: {warning.detail}"
            )

    return "\n".join(lines)
