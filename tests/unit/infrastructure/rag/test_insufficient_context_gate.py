"""The insufficient-context gate decision logic -- [M3-07].

Pure-logic tests here; the snapshot-driven test that enforces the M3 exit
criterion ("gate recall on the unanswerable subset 100%, by an automated test")
replays ``eval/insufficient_context_gate_signals.json`` and lives at the bottom
of this file.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from domain.clause_classification import ClauseType
from infrastructure.rag.insufficient_context_gate import (
    GateSignals,
    GateTrigger,
    MissingFactCategory,
    asks_for_instance_value,
    classify_missing_information,
    evaluate_gate,
    needs_verified_instance_value,
)

_REPO_ROOT = Path(__file__).resolve().parents[4]
_SNAPSHOT_PATH = _REPO_ROOT / "eval/insufficient_context_gate_signals.json"
_UNANSWERABLE_PATH = _REPO_ROOT / "data/golden_set/unanswerable.jsonl"


def _signals(
    *, top_score: float, n_returned: int = 10, clause_types: tuple[str, ...] = ()
) -> GateSignals:
    scores = (
        tuple(
            sorted(
                (top_score, *[max(0.0, top_score - 0.1)] * (n_returned - 1)),
                reverse=True,
            )
        )
        if n_returned
        else ()
    )
    return GateSignals(
        top_score=top_score if n_returned else 0.0,
        reranked_scores=scores,
        retrieved_clause_ids=tuple(f"1:c{i}" for i in range(n_returned)),
        retrieved_clause_types=tuple(ClauseType(t) for t in clause_types),
        k_requested=10,
        n_returned=n_returned,
    )


# A "rule" question about a deductible: names franquia but asks *how* it is
# structured -> not an instance-value question, only the primary floor applies.
_RULE_Q = "De que forma o valor da franquia é estruturado?"
# An "instance-value" question: names franquia and asks for the exact figure ->
# the strict INSTANCE_VALUE_TOP_SCORE_THRESHOLD applies.
_VALUE_Q = "Qual é o valor exato da franquia contratada para o veículo segurado?"


@pytest.mark.unit
def test_sufficient_when_score_clears_both_floors() -> None:
    result = evaluate_gate(
        _VALUE_Q,
        _signals(top_score=0.9),
        top_score_threshold=0.5,
        instance_value_threshold=0.84,
    )
    assert result.sufficient is True
    assert result.trigger is GateTrigger.NONE
    assert result.missing_category is None
    assert result.explanation == ""

    # A rule question only has to clear the primary floor; exactly at it is fine.
    assert evaluate_gate(
        _RULE_Q, _signals(top_score=0.5), top_score_threshold=0.5
    ).sufficient


@pytest.mark.unit
def test_low_relevance_abstention_has_a_structured_result() -> None:
    result = evaluate_gate(_RULE_Q, _signals(top_score=0.3), top_score_threshold=0.5)
    assert result.sufficient is False
    assert result.trigger is GateTrigger.LOW_RELEVANCE
    assert result.missing_category is MissingFactCategory.DEDUCTIBLE
    assert "franquia" in result.explanation
    assert "docs/SCOPE.md" in result.explanation
    assert result.top_score == pytest.approx(0.3)
    assert result.threshold == pytest.approx(0.5)
    assert result.closest_clause_ids == ("1:c0", "1:c1", "1:c2")


@pytest.mark.unit
def test_an_instance_value_question_abstains_between_the_two_floors() -> None:
    # Score 0.7: clears the primary floor (0.5) but not the strict floor (0.84),
    # and it asks for a specific figure -> abstain.
    result = evaluate_gate(
        _VALUE_Q,
        _signals(top_score=0.7),
        top_score_threshold=0.5,
        instance_value_threshold=0.84,
    )
    assert result.sufficient is False
    assert result.trigger is GateTrigger.UNVERIFIED_INSTANCE_VALUE
    assert result.missing_category is MissingFactCategory.DEDUCTIBLE
    assert "specific figure" in result.explanation

    # The same score for a *rule* question about the same fact is sufficient.
    assert evaluate_gate(
        _RULE_Q,
        _signals(top_score=0.7),
        top_score_threshold=0.5,
        instance_value_threshold=0.84,
    ).sufficient


@pytest.mark.unit
def test_abstains_when_nothing_was_retrieved() -> None:
    result = evaluate_gate(
        "Qual o prêmio comercial cobrado?", _signals(top_score=0.0, n_returned=0)
    )
    assert result.sufficient is False
    assert result.trigger is GateTrigger.NO_CONTEXT
    assert result.missing_category is MissingFactCategory.PREMIUM
    assert "no clause" in result.explanation
    assert result.closest_clause_ids == ()


@pytest.mark.unit
@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Qual é o valor exato da franquia contratada?", True),
        ("Qual foi o montante do prêmio adicional cobrado no endosso?", True),
        ("Em qual data as coberturas começaram a vigorar?", True),
        # A rule / manner / quantity / yes-no question is not an instance-value ask.
        ("De que forma o valor da franquia é estruturado?", False),
        ("Quantas franquias serão cobradas do segurado?", False),
        ("Como é calculada a devolução proporcional do prêmio?", False),
        ("Há cobertura para danos estéticos mediante prêmio adicional?", False),
        # No policy-instance fact named at all.
        ("O que caracteriza a apropriação indébita do veículo?", False),
    ],
)
def test_needs_verified_instance_value(question: str, expected: bool) -> None:
    assert needs_verified_instance_value(question) is expected


@pytest.mark.unit
def test_asks_for_instance_value_ignores_the_fact_category() -> None:
    # asks_for_instance_value is purely about phrasing; the SCOPE.md fact filter
    # is applied separately by needs_verified_instance_value.
    assert asks_for_instance_value("Qual é o valor exato a ser pago?") is True
    assert asks_for_instance_value("De que forma o valor é definido?") is False


@pytest.mark.unit
@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Qual é o valor da franquia obrigatória?", MissingFactCategory.DEDUCTIBLE),
        (
            "Qual a importância segurada contratada para a cobertura compreensiva?",
            MissingFactCategory.INSURED_AMOUNT,
        ),
        (
            "Qual é o Limite Máximo de Indenização (LMI) contratado?",
            MissingFactCategory.INSURED_AMOUNT,
        ),
        (
            "Qual o custo total do prêmio comercial cobrado?",
            MissingFactCategory.PREMIUM,
        ),
        (
            "Qual é a data de término de vigência das coberturas?",
            MissingFactCategory.POLICY_PERIOD,
        ),
        (
            "A partir de qual dia as coberturas começaram a vigorar?",
            MissingFactCategory.POLICY_PERIOD,
        ),
        # Endorsement outranks a premium/limit also named in the same question.
        (
            "Qual foi o valor do prêmio adicional cobrado no endosso de "
            "substituição de veículo?",
            MissingFactCategory.ENDORSEMENT,
        ),
        (
            "Qual é o novo limite máximo de garantia após o endosso?",
            MissingFactCategory.ENDORSEMENT,
        ),
        # No policy-instance keyword -> OTHER, still an honest abstention label.
        (
            "O que caracteriza a apropriação indébita do veículo?",
            MissingFactCategory.OTHER,
        ),
    ],
)
def test_classify_missing_information_priority_order(
    question: str, expected: MissingFactCategory
) -> None:
    assert classify_missing_information(question) is expected


@pytest.mark.unit
def test_every_golden_unanswerable_question_gets_a_named_category() -> None:
    """`classify_missing_information` covers all 23 -- no bare `OTHER` fallback."""
    unanswerable = _UNANSWERABLE_PATH.read_text(encoding="utf-8")
    for line in unanswerable.splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        category = classify_missing_information(record["question"])
        assert category is not MissingFactCategory.OTHER, record["question_id"]


# --- The M3 exit-criterion enforcement: recall == 1.0 on the 23 -------------
#
# Replays the committed per-question signal snapshot produced by
# `make eval-insufficient-context-gate` -- so this runs in `make check` with no
# Postgres, no GPU, no `embed` group. The eval-marked
# `tests/eval/test_insufficient_context_gate_baseline.py` re-derives the same
# signals live as a retrieval-drift guard.


def _load_snapshot() -> dict[str, Any]:
    if not _SNAPSHOT_PATH.exists():
        pytest.skip(
            f"{_SNAPSHOT_PATH.name} not built; "
            "run `make eval-insufficient-context-gate`"
        )
    loaded: dict[str, Any] = json.loads(_SNAPSHOT_PATH.read_text(encoding="utf-8"))
    return loaded


@pytest.mark.unit
def test_snapshot_matches_the_pinned_gate_contract() -> None:
    from infrastructure.rag.insufficient_context_config import config_fingerprint

    provenance = _load_snapshot()["provenance"]
    assert provenance["gate_config_fingerprint"] == config_fingerprint()


@pytest.mark.unit
def test_every_unanswerable_question_triggers_the_gate() -> None:
    rows = _load_snapshot()["questions"]

    tp = fp = fn = 0
    for row in rows:
        signals = GateSignals(
            top_score=row["top_score"],
            reranked_scores=tuple(row["reranked_scores"]),
            retrieved_clause_ids=tuple(row["retrieved_clause_ids"]),
            retrieved_clause_types=tuple(
                ClauseType(t) for t in row["retrieved_clause_types"]
            ),
            k_requested=row["k_requested"],
            n_returned=row["n_returned"],
        )
        abstained = not evaluate_gate(row["question"], signals).sufficient
        if row["is_unanswerable"] and abstained:
            tp += 1
        elif row["is_unanswerable"] and not abstained:
            fn += 1
        elif not row["is_unanswerable"] and abstained:
            fp += 1

    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    assert fn == 0, "the gate must abstain on every unanswerable question"
    assert recall == 1.0
    # Precision is published in docs/INSUFFICIENT_CONTEXT_GATE.md; this floor
    # just guards against a regression that keeps recall but wrecks precision.
    assert precision >= 0.60
