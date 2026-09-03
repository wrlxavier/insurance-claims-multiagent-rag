"""Shared scope constraint, prepended to every agent prompt.

Canonical statement: docs/SCOPE.md. This module exists so that statement
has exactly one machine-enforceable copy — every prompt built anywhere in
the agent graph includes SCOPE_PREAMBLE rather than restating the
constraint in its own words.
"""

SCOPE_PREAMBLE = """\
The documents you are given are general and special conditions of \
registered insurance products filed with SUSEP (Brazil's insurance \
regulator) — product templates, not individual insurance contracts. They \
contain no contracted coverages, insured amounts, deductibles, policy \
periods, or endorsements.

Given that, you may only assess whether a described event is consistent \
or inconsistent with the conditions of the registered product. You may \
not describe a real claim as approved, rejected, payable, or resolved, \
and you may not state or imply a real-world coverage outcome — the \
contract-level facts that determination requires are not available to \
you.

Every verdict you produce must be exactly one of: compatible, \
incompatible, insufficient_information. Use insufficient_information \
whenever the retrieved context does not settle the question, rather than \
guessing.\
"""


def with_scope_preamble(body: str) -> str:
    """Prepend ``SCOPE_PREAMBLE`` to a node prompt body.

    The one machine-checked way to satisfy this package's rule that every node
    prompt carries the scope constraint: ``build_<node>_prompt`` returns
    ``with_scope_preamble(...)`` rather than its own concatenation.
    """
    return f"{SCOPE_PREAMBLE}\n\n{body}"
