"""Domain layer.

Pure business entities, value objects, and invariants (Policy,
PolicyClause, Claim, Assessment, HumanDecision, Citation, Verdict,
DecisionOutcome; the SusepProcess and Cnpj value objects). Depends only on
the standard library. Must never import FastAPI, SQLAlchemy, Pydantic,
LangGraph, or LangChain.

M5-01 note: the DoD names the clause entity ``Clause``; it ships as
``PolicyClause`` (domain/policy_clause.py) so it does not shadow the
existing parse-tree ``Clause`` (domain/clause_tree.py). See
docs/ARCHITECTURE.md.
"""
