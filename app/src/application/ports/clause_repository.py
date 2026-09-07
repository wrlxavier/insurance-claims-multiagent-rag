"""Port for reading clauses of registered products [M5-02].

Read-only. The clause corpus is reference data -- SUSEP's registered *condições
gerais* -- built by the M1 parsing pipeline and never written by the assessment
use cases, so this repository stands apart from the transactional
``UnitOfWork`` (which owns only ``AssessmentRepository``).

Its one M5-02 caller is ``SubmitHumanDecision``: when a reviewer edits an
assessment, every clause the edit cites must resolve to a real
``PolicyClause`` here, or the decision is rejected before the graph is resumed.
[M5-03] backs it with the ``chunk`` table; [M5-04] may grow it a read endpoint.
"""

from collections.abc import Sequence
from typing import Protocol

from domain.policy_clause import PolicyClause
from domain.susep_process import SusepProcess


class ClauseRepository(Protocol):
    """Look up clauses of registered products by id or by product."""

    def get(self, clause_id: str) -> PolicyClause | None:
        """Return the clause with ``clause_id``, or ``None`` if the corpus has none."""
        ...

    def get_many(self, clause_ids: Sequence[str]) -> tuple[PolicyClause, ...]:
        """Return the clauses that exist, in the order their ids were given.

        Ids with no matching clause are omitted -- the caller compares the
        returned count (or ids) against the request to find the gaps.
        """
        ...

    def list_for_policy(self, policy: SusepProcess) -> tuple[PolicyClause, ...]:
        """Return every clause of the product identified by ``policy``."""
        ...
