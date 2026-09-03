"""The CI citation-coverage gate, on fixtures [M4-10].

This is the DoD's "automated check: 100% of assertions carry a clause id, and
every id exists in the corpus. CI fails otherwise". The gate itself runs in CI
as ``make validate-citation-coverage`` against the committed snapshot and the
real corpus; these tests pin its *logic* offline, and in particular pin every
rejection case -- a gate nobody has watched fail is not evidence.
"""

import json
from pathlib import Path
from typing import Any

import pytest
from scripts.validate_citation_coverage import (
    CitationCoverageError,
    check_assertion_clause_ids_exist,
    check_every_assertion_carries_a_clause_id,
    check_recommendation_citations,
    load_snapshot,
)

_CORPUS = {"1:a", "1:b", "2:c"}


def _claim(**overrides: Any) -> dict[str, Any]:
    claim: dict[str, Any] = {
        "claim_id": "compatible-001",
        "cohort": "compatible",
        "compatibility_verdict": "compatible",
        "assertions": [{"statement": "O evento é coberto.", "clause_ids": ["1:a"]}],
        "recommendation_citation_ids": ["1:a"],
        "retrieved_clause_ids": ["1:a", "1:b"],
    }
    claim.update(overrides)
    return claim


# --- the happy path --------------------------------------------------------


@pytest.mark.unit
def test_a_well_formed_run_passes_every_check() -> None:
    claims = [_claim()]
    assert check_every_assertion_carries_a_clause_id(claims) == 1
    check_assertion_clause_ids_exist(claims, _CORPUS)
    check_recommendation_citations(claims, _CORPUS)


@pytest.mark.unit
def test_the_assertion_count_is_the_reported_denominator() -> None:
    claims = [
        _claim(
            assertions=[
                {"statement": "s1", "clause_ids": ["1:a"]},
                {"statement": "s2", "clause_ids": ["1:b"]},
            ]
        ),
        _claim(claim_id="incompatible-001", compatibility_verdict="incompatible"),
    ]
    assert check_every_assertion_carries_a_clause_id(claims) == 3


# --- abstentions are excluded, not failed ----------------------------------


@pytest.mark.unit
def test_an_abstention_with_no_assertions_is_not_a_violation() -> None:
    # `_abstain` writes free prose and no ids by design; failing it here would
    # flag the node for behaving correctly.
    claims = [
        _claim(
            compatibility_verdict="insufficient_information",
            assertions=[],
            recommendation_citation_ids=[],
        )
    ]
    assert check_every_assertion_carries_a_clause_id(claims) == 0


@pytest.mark.unit
def test_an_unassessed_claim_is_not_a_violation() -> None:
    # clarification_exhausted / retrieval-miss paths never run compatibility.
    claims = [
        _claim(
            compatibility_verdict=None,
            assertions=[],
            recommendation_citation_ids=[],
            retrieved_clause_ids=[],
        )
    ]
    assert check_every_assertion_carries_a_clause_id(claims) == 0


# --- the rejections --------------------------------------------------------


@pytest.mark.unit
def test_an_assertion_without_a_clause_id_fails() -> None:
    claims = [_claim(assertions=[{"statement": "s", "clause_ids": []}])]
    with pytest.raises(CitationCoverageError, match="carries no clause id"):
        check_every_assertion_carries_a_clause_id(claims)


@pytest.mark.unit
def test_a_settled_verdict_with_no_assertions_at_all_fails() -> None:
    claims = [_claim(assertions=[])]
    with pytest.raises(CitationCoverageError, match="no grounded assertions"):
        check_every_assertion_carries_a_clause_id(claims)


@pytest.mark.unit
def test_a_cited_id_absent_from_the_corpus_fails() -> None:
    claims = [_claim(assertions=[{"statement": "s", "clause_ids": ["9:ghost"]}])]
    with pytest.raises(CitationCoverageError, match="absent from the corpus"):
        check_assertion_clause_ids_exist(claims, _CORPUS)


@pytest.mark.unit
def test_a_recommendation_citation_absent_from_the_corpus_fails() -> None:
    claims = [_claim(recommendation_citation_ids=["9:ghost"])]
    with pytest.raises(CitationCoverageError, match="absent from"):
        check_recommendation_citations(claims, _CORPUS)


@pytest.mark.unit
def test_a_recommendation_citation_retrieval_never_returned_fails() -> None:
    # The [M4-08] "never introduce a citation no upstream node produced"
    # guarantee, re-checked on a real run rather than on a fake.
    claims = [_claim(recommendation_citation_ids=["2:c"])]
    with pytest.raises(CitationCoverageError, match="retrieval never returned"):
        check_recommendation_citations(claims, _CORPUS)


# --- missing evidence is an error, never a skip ----------------------------


@pytest.mark.unit
def test_a_missing_snapshot_raises_rather_than_passing_quietly(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError, match="make eval-end-to-end"):
        load_snapshot(tmp_path / "absent.json")


@pytest.mark.unit
def test_an_empty_snapshot_raises(tmp_path: Path) -> None:
    path = tmp_path / "snapshot.json"
    path.write_text(json.dumps({"provenance": {}, "claims": []}), encoding="utf-8")
    with pytest.raises(CitationCoverageError, match="no claims"):
        load_snapshot(path)


@pytest.mark.unit
def test_a_present_snapshot_loads(tmp_path: Path) -> None:
    path = tmp_path / "snapshot.json"
    path.write_text(
        json.dumps({"provenance": {"generated_at_utc": "x"}, "claims": [_claim()]}),
        encoding="utf-8",
    )
    assert len(load_snapshot(path)["claims"]) == 1
