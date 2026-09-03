"""Application layer.

Use cases and ports (interfaces) that orchestrate domain rules. Depends
only on domain. Has no knowledge of infrastructure details (LangGraph,
database, HTTP) — only the ports that represent them.

M5-02 adds the claim-assessment surface: the ``ClauseRepository``,
``AssessmentRepository``, ``ClaimAssessmentOrchestrator``, ``UnitOfWork``
and ``Clock`` ports, the ``SubmitClaim`` / ``GetAssessment`` /
``SubmitHumanDecision`` / ``ListAssessments`` use cases, and the
``AssessmentRecord`` / ``OrchestratorResult`` DTOs the layer speaks in.
``ClaimAssessmentOrchestrator`` hides LangGraph completely — nothing here
knows a graph produces the assessment. Enforced by
tests/architecture/test_layer_boundaries.py.
"""
