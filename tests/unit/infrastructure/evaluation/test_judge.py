"""The faithfulness / context-relevance judge, with a fake chain [M4-10].

No network: every test drives ``judge_*`` through a stub model whose
``with_structured_output`` returns a scripted sequence of batches. What is
pinned here is the arithmetic and the failure handling -- the majority vote,
the per-pass record that makes the variance reportable, and what happens when a
pass returns fewer judgments than it was asked for.
"""

from collections.abc import Sequence
from typing import Any

import pytest

from infrastructure.evaluation.judge import (
    AssertionJudgment,
    ClauseRelevanceJudgment,
    ContextRelevanceBatch,
    FaithfulnessBatch,
    build_context_relevance_prompt,
    build_faithfulness_prompt,
    invoke_judge,
    judge_context_relevance,
    judge_faithfulness,
    majority_of,
)


class _FakeChain:
    def __init__(self, batches: list[Any]) -> None:
        self._batches = list(batches)
        self.prompts: list[str] = []

    def invoke(self, prompt: str) -> Any:
        self.prompts.append(prompt)
        result = self._batches.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class _FakeModel:
    def __init__(self, batches: list[Any]) -> None:
        self.chain = _FakeChain(batches)

    def with_structured_output(self, schema: type, **kwargs: Any) -> _FakeChain:
        return self.chain


def _faith(*levels: str) -> FaithfulnessBatch:
    return FaithfulnessBatch(
        judgments=[
            AssertionJudgment(index=i, support=level, rationale="porque sim")
            for i, level in enumerate(levels, start=1)
        ]
    )


def _relevance(*pairs: tuple[str, bool]) -> ContextRelevanceBatch:
    return ContextRelevanceBatch(
        judgments=[
            ClauseRelevanceJudgment(clause_id=cid, relevant=rel, rationale="porque sim")
            for cid, rel in pairs
        ]
    )


def _model(batches: list[Any]) -> Any:
    return _FakeModel(batches)


# --- the majority vote -----------------------------------------------------


@pytest.mark.unit
def test_majority_picks_the_most_common_label() -> None:
    aggregate = majority_of(["supported", "unsupported", "supported"])
    assert aggregate.majority == "supported"
    assert aggregate.n_passes == 3
    assert not aggregate.unanimous


@pytest.mark.unit
def test_unanimity_is_reported_not_assumed() -> None:
    assert majority_of(["relevant"] * 3).unanimous is True


@pytest.mark.unit
def test_a_tie_breaks_toward_the_first_pass_deterministically() -> None:
    # Unreachable at JUDGE_PASSES=3, but a caller overriding the count must
    # still get a defined answer rather than dict ordering.
    assert majority_of(["supported", "unsupported"]).majority == "supported"
    assert majority_of(["unsupported", "supported"]).majority == "unsupported"


@pytest.mark.unit
def test_majority_of_nothing_raises() -> None:
    with pytest.raises(ValueError, match="at least one pass"):
        majority_of([])


# --- faithfulness ----------------------------------------------------------


@pytest.mark.unit
def test_faithfulness_runs_every_pass_and_keeps_each_pass_value() -> None:
    model = _model(
        [
            _faith("supported", "unsupported"),
            _faith("supported", "supported"),
            _faith("supported", "unsupported"),
        ]
    )
    results = judge_faithfulness(
        model,
        "relato",
        [("afirmação 1", ["1:a"]), ("afirmação 2", ["1:b"])],
        {"1:a": "texto a", "1:b": "texto b"},
        passes=3,
    )
    assert [r.majority for r in results] == ["supported", "unsupported"]
    assert results[0].unanimous is True
    assert results[1].pass_values == ("unsupported", "supported", "unsupported")


@pytest.mark.unit
def test_a_skipped_judgment_counts_as_unsupported_and_shows_as_disagreement() -> None:
    # The conservative direction for a *faithfulness* rate, and visible in
    # pass_values rather than silently dropped from the denominator.
    model = _model([_faith("supported", "supported"), _faith("supported")])
    results = judge_faithfulness(
        model,
        "relato",
        [("a1", ["1:a"]), ("a2", ["1:b"])],
        {"1:a": "x", "1:b": "y"},
        passes=2,
    )
    assert results[1].pass_values == ("supported", "unsupported")


@pytest.mark.unit
def test_no_assertions_costs_no_call() -> None:
    model = _model([])
    assert judge_faithfulness(model, "relato", [], {}, passes=3) == []
    assert model.chain.prompts == []


# --- context relevance -----------------------------------------------------


@pytest.mark.unit
def test_context_relevance_is_keyed_by_clause_id() -> None:
    model = _model(
        [
            _relevance(("1:a", True), ("1:b", False)),
            _relevance(("1:a", True), ("1:b", True)),
            _relevance(("1:a", True), ("1:b", False)),
        ]
    )
    results = judge_context_relevance(
        model, "relato", [("1:a", "texto a"), ("1:b", "texto b")], passes=3
    )
    assert results["1:a"].majority == "relevant"
    assert results["1:b"].majority == "irrelevant"
    assert results["1:b"].unanimous is False


@pytest.mark.unit
def test_a_clause_the_judge_omitted_counts_as_irrelevant() -> None:
    model = _model([_relevance(("1:a", True))])
    results = judge_context_relevance(
        model, "relato", [("1:a", "x"), ("1:b", "y")], passes=1
    )
    assert results["1:b"].majority == "irrelevant"


# --- retries ---------------------------------------------------------------


@pytest.mark.unit
def test_a_transient_failure_is_retried_then_succeeds() -> None:
    slept: list[float] = []
    model = _model([RuntimeError("502"), _faith("supported")])
    batch = invoke_judge(
        model,
        "prompt",
        FaithfulnessBatch,
        max_attempts=3,
        delay_seconds=0.0,
        sleep=slept.append,
    )
    assert batch.judgments[0].support == "supported"
    assert slept == [0.0]


@pytest.mark.unit
def test_a_persistent_failure_re_raises_rather_than_defaulting() -> None:
    # There is no sane fallback value for a judgment: a defaulted one would
    # enter a published average unnoticed.
    model = _model([RuntimeError("boom")] * 3)
    with pytest.raises(RuntimeError, match="boom"):
        invoke_judge(
            model,
            "prompt",
            FaithfulnessBatch,
            max_attempts=3,
            delay_seconds=0.0,
            sleep=lambda _: None,
        )


# --- the committed prompts -------------------------------------------------


@pytest.mark.unit
def test_the_faithfulness_prompt_shows_only_the_cited_clauses() -> None:
    prompt = build_faithfulness_prompt(
        "bati o carro",
        [("o evento é uma colisão", ["1:a"])],
        {"1:a": "texto da cláusula A", "1:z": "texto da cláusula Z"},
    )
    assert "texto da cláusula A" in prompt
    assert "texto da cláusula Z" not in prompt, "an uncited clause is not evidence"
    assert "bati o carro" in prompt
    assert "Retorne exatamente 1 avaliação(ões)." in prompt


@pytest.mark.unit
def test_the_faithfulness_prompt_names_the_grounded_not_correct_distinction() -> None:
    prompt = build_faithfulness_prompt("relato", [("a", ["1:a"])], {"1:a": "t"})
    assert "Estar certo não é o mesmo que estar fundamentado" in prompt


@pytest.mark.unit
def test_a_missing_excerpt_is_marked_not_silently_empty() -> None:
    prompt = build_faithfulness_prompt("relato", [("a", ["1:missing"])], {})
    assert "(texto indisponível)" in prompt


@pytest.mark.unit
def test_the_relevance_prompt_lists_every_clause_with_its_id() -> None:
    clauses: Sequence[tuple[str, str]] = [("1:a", "texto a"), ("1:b", "texto b")]
    prompt = build_context_relevance_prompt("relato", clauses)
    for clause_id, text in clauses:
        assert f"[{clause_id}]" in prompt
        assert text in prompt
    assert "Retorne exatamente 2 avaliação(ões)." in prompt


@pytest.mark.unit
def test_the_relevance_prompt_says_an_exclusion_is_relevant() -> None:
    # The single instruction most likely to be got wrong by a naive judge:
    # a clause that denies the claim is still a clause that bears on it.
    prompt = build_context_relevance_prompt("relato", [("1:a", "t")])
    assert "Uma cláusula que EXCLUI o evento é relevante" in prompt
