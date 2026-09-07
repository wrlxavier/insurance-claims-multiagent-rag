"""The SQLAlchemy ``ClauseRepository`` -- [M5-03].

Implements [application.ports.clause_repository.ClauseRepository] over the
existing ``chunk`` table -- there is no separate ``clause`` table, and the port
docstring already commits to this backing. Read-only: the clause corpus is
reference data built by the M1 pipeline, outside the assessment transaction.

A ``chunk`` row records every clause-tree id folded into it in
``source_clause_ids`` (the anchor plus any merged siblings), and a long clause
splits across several rows. So a clause is reassembled by collecting every row
whose ``source_clause_ids`` contains the wanted id, ordering by ``chunk_index``
and joining ``display_text`` -- the same reconstruction
[infrastructure.rag.graph_retrieval_adapter.build_clause_index] does for a
retrieval hit, which is where a [domain.citation.Citation]'s ``clause_id`` comes
from.
"""

from collections.abc import Sequence

from sqlalchemy import any_, select
from sqlalchemy.orm import Session

from domain.clause_classification import ClauseType
from domain.policy_clause import PolicyClause
from domain.susep_process import SusepProcess
from infrastructure.database.models import ChunkRow

_TEXT_JOIN = "\n\n"


class SqlAlchemyClauseRepository:
    """Look up registered-product clauses, projected from the ``chunk`` table."""

    def __init__(self, session: Session) -> None:
        """Bind the repository to a session (reads only; no transaction needed)."""
        self._session = session

    def get(self, clause_id: str) -> PolicyClause | None:
        """Return the clause with ``clause_id``, or ``None`` if the corpus has none."""
        rows = list(
            self._session.execute(
                select(ChunkRow).where(any_(ChunkRow.source_clause_ids) == clause_id)
            ).scalars()
        )
        return _assemble(clause_id, rows)

    def get_many(self, clause_ids: Sequence[str]) -> tuple[PolicyClause, ...]:
        """Return the clauses that exist, in the order their ids were given.

        Its one caller (``SubmitHumanDecision`` validating an edit's citations)
        passes a handful of ids, so a lookup per id is fine and keeps the query
        simple.
        """
        found = (self.get(clause_id) for clause_id in clause_ids)
        return tuple(clause for clause in found if clause is not None)

    def list_for_policy(self, policy: SusepProcess) -> tuple[PolicyClause, ...]:
        """Return every clause of the product identified by ``policy``."""
        rows = list(
            self._session.execute(
                select(ChunkRow).where(ChunkRow.susep_process == policy.value)
            ).scalars()
        )

        by_id: dict[str, list[ChunkRow]] = {}
        for row in rows:
            for source_id in row.source_clause_ids:
                by_id.setdefault(source_id, []).append(row)

        assembled = (
            _assemble(clause_id, chunk_rows) for clause_id, chunk_rows in by_id.items()
        )
        return tuple(clause for clause in assembled if clause is not None)


def _assemble(clause_id: str, rows: Sequence[ChunkRow]) -> PolicyClause | None:
    """Build one ``PolicyClause`` from the chunk rows that carry ``clause_id``."""
    if not rows:
        return None
    ordered = sorted(rows, key=lambda row: row.chunk_index)
    anchor = ordered[0]
    text = _TEXT_JOIN.join(row.display_text for row in ordered)
    return PolicyClause(
        clause_id=clause_id,
        susep_process=SusepProcess(anchor.susep_process),
        document_id=anchor.document_id,
        clause_type=ClauseType(anchor.clause_type),
        text=text,
    )
