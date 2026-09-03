"""The project's one verdict vocabulary [M0-06].

Canonical string source: ``SCOPE_PREAMBLE`` in
[infrastructure.graph.prompts.scope_preamble] -- "Every verdict you produce
must be exactly one of: compatible, incompatible, insufficient_information.
Use insufficient_information whenever the retrieved context does not settle
the question, rather than guessing."

A domain value object so every layer shares one definition rather than
restating it: the agent-graph state ([M4-01]), the evaluation schemas
([infrastructure.evaluation.golden_set_schema.ExpectedVerdict] is now an
alias of this), and M5-01's domain entities (an assessment's verdict is one
of these three). tests/architecture/test_scope_vocabulary.py guards this
file against the forbidden contract-outcome vocabulary.
"""

from enum import Enum


class Verdict(Enum):
    """The only three assessment outcomes the project permits.

    Never the contract-outcome vocabulary M0-06 forbids: the corpus is
    registered product conditions, not contracts, so the system judges
    consistency with a product, never a real-world claim result
    (docs/SCOPE.md). tests/architecture/test_scope_vocabulary.py enforces it.
    """

    COMPATIBLE = "compatible"
    INCOMPATIBLE = "incompatible"
    INSUFFICIENT_INFORMATION = "insufficient_information"
