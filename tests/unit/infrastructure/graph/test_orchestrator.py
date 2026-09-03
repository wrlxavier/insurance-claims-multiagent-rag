"""The LangGraph orchestrator adapter's framework-free seams [M5-04].

The compile/invoke/checkpointer path needs a real Postgres and is covered by
``tests/integration/test_assessment_api.py``. Here: the policy-header prepend and
the capturing sink, which are pure.
"""

from datetime import UTC, datetime

import pytest

from domain.claim import Claim
from domain.susep_process import SusepProcess
from infrastructure.graph.orchestrator import (
    LangGraphClaimAssessmentOrchestrator,
    _CapturingAuditSink,
)
from infrastructure.graph.state import AuditEvent, AuditRecord

_NOW = datetime(2026, 9, 3, tzinfo=UTC)


@pytest.mark.unit
def test_claim_text_is_the_narrative_when_no_policy_ref() -> None:
    claim = Claim(claim_id="c1", raw_text="Bati o carro.", submitted_at=_NOW)
    assert LangGraphClaimAssessmentOrchestrator._claim_text(claim) == "Bati o carro."


@pytest.mark.unit
def test_claim_text_prepends_the_registered_policy_header() -> None:
    claim = Claim(
        claim_id="c1",
        raw_text="Bati o carro.",
        submitted_at=_NOW,
        policy_ref=SusepProcess("15414.610650/2024-59"),
    )

    text = LangGraphClaimAssessmentOrchestrator._claim_text(claim)

    assert text == (
        "[Apólice registrada: processo SUSEP 15414.610650/2024-59]\nBati o carro."
    )


@pytest.mark.unit
def test_capturing_sink_keeps_the_last_full_trail() -> None:
    sink = _CapturingAuditSink()
    records = [AuditRecord(AuditEvent(node="retrieval", action="retrieve"))]

    written = sink.record(claim_id="c1", thread_id="a1", records=records)

    assert written == 1
    assert sink.records == records
    assert sink.calls == [("c1", "a1")]
