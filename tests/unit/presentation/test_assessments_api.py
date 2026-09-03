"""Endpoint behaviour and the error -> HTTP mapping -- [M5-04].

Driven through ``tests/unit/presentation/conftest.py``'s ``TestClient`` wired to
the in-memory application fakes: no Postgres, no graph, no LLM.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from application.assessment_record import AssessmentRecord, AssessmentStatus
from domain.citation import Citation
from domain.clause_classification import ClauseType
from domain.human_decision import DecisionOutcome, HumanDecision
from domain.susep_process import SusepProcess
from domain.verdict import Verdict
from tests.unit.application.fakes import (
    FIXED_NOW,
    SUSEP,
    FakeClaimAssessmentOrchestrator,
    make_audit_entry,
    make_citation,
    make_orchestrator_result,
    make_policy_clause,
    make_record,
)
from tests.unit.presentation.conftest import Harness

_CANONICAL_SUSEP = "15414.610650/2024-59"


# --------------------------------------------------------------------------- #
# POST /v1/assessments
# --------------------------------------------------------------------------- #


def test_submit_claim_returns_202_with_id_and_location(harness: Harness) -> None:
    response = harness.client.post(
        "/v1/assessments", json={"raw_text": "Bati o carro."}
    )

    assert response.status_code == 202
    body = response.json()
    assessment_id = body["assessment_id"]
    assert body["status"] == "awaiting_review"
    assert response.headers["location"] == f"/v1/assessments/{assessment_id}"
    stored = harness.store[assessment_id]
    assert stored.status is AssessmentStatus.AWAITING_REVIEW


def test_submit_claim_empty_narrative_is_422_validation(harness: Harness) -> None:
    response = harness.client.post("/v1/assessments", json={"raw_text": ""})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "request_validation_error"


def test_submit_claim_bad_policy_ref_is_422(harness: Harness) -> None:
    response = harness.client.post(
        "/v1/assessments", json={"raw_text": "x", "policy_ref": "not-a-susep"}
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_susep_process"


def test_submit_claim_threads_policy_ref_into_the_claim(harness: Harness) -> None:
    harness.client.post(
        "/v1/assessments", json={"raw_text": "x", "policy_ref": _CANONICAL_SUSEP}
    )

    _, claim = harness.orchestrator.started[0]
    assert claim.policy_ref == SusepProcess(_CANONICAL_SUSEP)


def test_submit_claim_orchestrator_not_pausing_is_502(harness: Harness) -> None:
    harness.orchestrator = FakeClaimAssessmentOrchestrator(
        start_result=make_orchestrator_result(awaiting_review=False)
    )

    response = harness.client.post("/v1/assessments", json={"raw_text": "x"})

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "orchestrator_contract_error"


# --------------------------------------------------------------------------- #
# GET /v1/assessments/{id}  +  GET /v1/assessments
# --------------------------------------------------------------------------- #


def test_get_assessment_returns_structured_citations(harness: Harness) -> None:
    harness.store["a1"] = make_record(assessment_id="a1")

    body = harness.client.get("/v1/assessments/a1").json()

    assert body["verdict"] == "compatible"
    assert body["is_grounded"] is True
    citation = body["citations"][0]
    assert set(citation) == {
        "clause_id",
        "document_id",
        "susep_process",
        "clause_type",
        "excerpt",
        "relevance_score",
    }
    assert body["decision"] is None


def test_get_assessment_abstain_record_has_empty_citations(harness: Harness) -> None:
    harness.store["a1"] = make_record(
        assessment_id="a1",
        citations=(),
        verdict=Verdict.INSUFFICIENT_INFORMATION,
    )

    body = harness.client.get("/v1/assessments/a1").json()

    assert body["citations"] == []
    assert body["is_grounded"] is False


def test_get_unknown_assessment_is_404(harness: Harness) -> None:
    response = harness.client.get("/v1/assessments/nope")

    assert response.status_code == 404
    error = response.json()["error"]
    assert error["code"] == "assessment_not_found"
    assert error["details"] == {"assessment_id": "nope"}


def test_list_assessments_newest_first_with_filters(harness: Harness) -> None:
    harness.store["old"] = make_record(
        assessment_id="old", claim_id="c1", created_at=FIXED_NOW - timedelta(hours=1)
    )
    harness.store["new"] = make_record(
        assessment_id="new", claim_id="c1", created_at=FIXED_NOW
    )
    harness.store["other"] = make_record(assessment_id="other", claim_id="c2")

    rows = harness.client.get("/v1/assessments", params={"claim_id": "c1"}).json()

    assert [r["assessment_id"] for r in rows] == ["new", "old"]


def test_list_assessments_rejects_bad_paging(harness: Harness) -> None:
    assert harness.client.get("/v1/assessments", params={"limit": 0}).status_code == 422
    assert (
        harness.client.get("/v1/assessments", params={"offset": -1}).status_code == 422
    )


# --------------------------------------------------------------------------- #
# POST /v1/assessments/{id}/decision
# --------------------------------------------------------------------------- #


def _awaiting(assessment_id: str = "a1", **overrides: object) -> AssessmentRecord:
    return make_record(assessment_id=assessment_id, **overrides)


def test_decision_approve_settles_the_record(harness: Harness) -> None:
    harness.store["a1"] = _awaiting()

    body = harness.client.post(
        "/v1/assessments/a1/decision",
        json={"decision": "approve", "notes": "conferido"},
    ).json()

    assert body["status"] == "decided"
    assert body["decision"]["decision"] == "approve"
    assert body["decision"]["notes"] == "conferido"
    assert harness.orchestrator.resumed[0][0] == "a1"
    # system opinion unchanged
    assert body["verdict"] == "compatible"


def test_decision_reject_settles_the_record(harness: Harness) -> None:
    harness.store["a1"] = _awaiting()

    body = harness.client.post(
        "/v1/assessments/a1/decision", json={"decision": "reject"}
    ).json()

    assert body["decision"]["decision"] == "reject"


def test_decision_edit_records_the_edited_assessment(harness: Harness) -> None:
    harness.store["a1"] = _awaiting()
    harness.clauses["c:1"] = make_policy_clause(clause_id="c:1")

    payload = {
        "decision": "edit",
        "edited": {
            "verdict": "incompatible",
            "reasoning": "Exclusão aplicável.",
            "recommended_action": "Negar.",
            "confidence": 0.6,
            "citations": [
                {
                    "clause_id": "c:1",
                    "document_id": "doc",
                    "susep_process": _CANONICAL_SUSEP,
                    "clause_type": "exclusion",
                    "excerpt": "trecho",
                }
            ],
        },
    }
    body = harness.client.post("/v1/assessments/a1/decision", json=payload).json()

    assert body["decision"]["edited_assessment"]["verdict"] == "incompatible"
    # the system's own verdict is untouched
    assert body["verdict"] == "compatible"


def test_decision_edit_without_payload_is_422(harness: Harness) -> None:
    harness.store["a1"] = _awaiting()

    response = harness.client.post(
        "/v1/assessments/a1/decision", json={"decision": "edit"}
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"


def test_decision_non_edit_carrying_payload_is_422(harness: Harness) -> None:
    harness.store["a1"] = _awaiting()

    response = harness.client.post(
        "/v1/assessments/a1/decision",
        json={
            "decision": "approve",
            "edited": {
                "verdict": "compatible",
                "reasoning": "x",
                "recommended_action": "y",
                "confidence": 0.5,
                "citations": [],
            },
        },
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"


def test_decision_edit_citing_unknown_clause_is_422(harness: Harness) -> None:
    harness.store["a1"] = _awaiting()

    response = harness.client.post(
        "/v1/assessments/a1/decision",
        json={
            "decision": "edit",
            "edited": {
                "verdict": "incompatible",
                "reasoning": "x",
                "recommended_action": "y",
                "confidence": 0.5,
                "citations": [
                    {
                        "clause_id": "ghost:9",
                        "document_id": "doc",
                        "susep_process": _CANONICAL_SUSEP,
                        "clause_type": "exclusion",
                        "excerpt": "t",
                    }
                ],
            },
        },
    )

    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "unknown_clause"
    assert error["details"] == {"clause_ids": ["ghost:9"]}


def test_decision_edit_citing_nothing_is_422_citation_required(
    harness: Harness,
) -> None:
    harness.store["a1"] = _awaiting()

    response = harness.client.post(
        "/v1/assessments/a1/decision",
        json={
            "decision": "edit",
            "edited": {
                "verdict": "incompatible",
                "reasoning": "x",
                "recommended_action": "y",
                "confidence": 0.5,
                "citations": [],
            },
        },
    )

    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "citation_required"
    assert error["details"] == {"assessment_id": "a1"}


def test_decision_on_decided_assessment_is_409(harness: Harness) -> None:
    decision = HumanDecision(
        assessment_id="a1",
        decision=DecisionOutcome.APPROVE,
        decided_at=FIXED_NOW,
    )
    harness.store["a1"] = make_record(
        assessment_id="a1", status=AssessmentStatus.DECIDED, decision=decision
    )

    response = harness.client.post(
        "/v1/assessments/a1/decision", json={"decision": "approve"}
    )

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "assessment_already_decided"


def test_decision_on_unknown_assessment_is_404(harness: Harness) -> None:
    response = harness.client.post(
        "/v1/assessments/ghost/decision", json={"decision": "approve"}
    )

    assert response.status_code == 404


def test_decision_orchestrator_not_finishing_is_502(harness: Harness) -> None:
    harness.store["a1"] = _awaiting()
    harness.orchestrator = FakeClaimAssessmentOrchestrator(
        resume_result=make_orchestrator_result(awaiting_review=True)
    )

    response = harness.client.post(
        "/v1/assessments/a1/decision", json={"decision": "approve"}
    )

    assert response.status_code == 502


# --------------------------------------------------------------------------- #
# GET /v1/assessments/{id}/audit
# --------------------------------------------------------------------------- #


def test_audit_unknown_assessment_is_404(harness: Harness) -> None:
    assert harness.client.get("/v1/assessments/ghost/audit").status_code == 404


def test_audit_empty_before_a_decision(harness: Harness) -> None:
    harness.store["a1"] = _awaiting()

    body = harness.client.get("/v1/assessments/a1/audit").json()

    assert body == {"assessment_id": "a1", "entries": []}


def test_audit_returns_the_trail_in_sequence_order(harness: Harness) -> None:
    harness.store["a1"] = _awaiting()
    harness.audit_store["a1"] = [
        make_audit_entry(
            sequence=1,
            node="human_review",
            action="human_decision:approve",
            payload={"decision": "approve"},
        ),
        make_audit_entry(sequence=0, node="retrieval", action="retrieve_clauses"),
    ]

    body = harness.client.get("/v1/assessments/a1/audit").json()

    assert [e["sequence"] for e in body["entries"]] == [0, 1]
    assert body["entries"][-1]["node"] == "human_review"
    assert body["entries"][-1]["payload"] == {"decision": "approve"}


# --------------------------------------------------------------------------- #
# the error envelope / catch-all
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("path", ["/v1/assessments/x", "/v1/assessments/x/audit"])
def test_error_envelope_shape(harness: Harness, path: str) -> None:
    body = harness.client.get(path).json()

    assert set(body) == {"error"}
    assert set(body["error"]) == {"code", "message", "details"}


def test_unexpected_error_is_logged_500_without_traceback(harness: Harness) -> None:
    boom = RuntimeError("kaboom")
    harness.orchestrator = FakeClaimAssessmentOrchestrator(raise_on_start=boom)

    response = harness.client.post("/v1/assessments", json={"raw_text": "x"})

    assert response.status_code == 500
    body = response.json()
    assert body["error"]["code"] == "internal_error"
    assert "kaboom" not in body["error"]["message"]


def test_health_is_ok(harness: Harness) -> None:
    assert harness.client.get("/health").json() == {"status": "ok"}


def test_citation_value_objects_render_as_canonical_strings(harness: Harness) -> None:
    harness.store["a1"] = make_record(
        assessment_id="a1",
        citations=(
            make_citation(
                clause_id="c:2",
                susep_process=SUSEP,
                clause_type=ClauseType.EXCLUSION,
            ),
        ),
    )

    citation = harness.client.get("/v1/assessments/a1").json()["citations"][0]

    assert citation["susep_process"] == SUSEP.value
    assert citation["clause_type"] == "exclusion"
    # sanity: the value object round-trips
    assert Citation(
        clause_id="c:2",
        document_id="d",
        susep_process=SusepProcess(citation["susep_process"]),
        clause_type=ClauseType(citation["clause_type"]),
        excerpt="e",
    )
