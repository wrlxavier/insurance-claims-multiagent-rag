"""Ports (interfaces) that infrastructure adapters implement.

M5-02: ``clause_repository``, ``assessment_repository``,
``claim_assessment_orchestrator``, ``unit_of_work`` and ``clock`` -- the
contracts the assessment use cases depend on. Each speaks domain (or
application DTO) types only; ``claim_assessment_orchestrator`` in particular
exposes nothing of the LangGraph run behind it.
"""
