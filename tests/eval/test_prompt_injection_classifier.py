"""Live domain benchmark for the optional classifier ([M5-08 Appendix]).

The unit tests (``tests/unit/infrastructure/guardrails/``) cover the wiring
with a faked ``transformers``, which cannot demonstrate anything about the
real model's actual false-positive/detection behaviour. This eval-marked
test is the on-demand run of the Appendix's own ask ("measure false-positive
rates ... and evaluate per-node latency"). Unlike
``tests/eval/test_prompt_injection.py``, there is no pass/fail here -- the
measured numbers themselves are the result, written up in
``docs/PROMPT_INJECTION_CLASSIFIER.md``. This test only asserts the run
completes and produces a row per fixture, so a broken wiring still fails
loudly.

Needs the optional ``embed`` uv group (``uv sync --group embed`` /
``uv run --group embed``). No LLM, no Postgres. Skips cleanly when
``transformers`` is not importable, so ``make test-eval`` stays green (and
free) on a machine that never opted into the group.
"""

import importlib.util

import pytest

pytestmark = pytest.mark.eval


def _skip_unless_ready() -> None:
    if importlib.util.find_spec("transformers") is None:
        pytest.skip(
            "transformers not installed; run `uv sync --group embed` then "
            "`make eval-prompt-injection-classifier` for real"
        )


@pytest.mark.eval
def test_the_benchmark_scores_every_benign_and_adversarial_row() -> None:
    _skip_unless_ready()

    from scripts.eval_prompt_injection_classifier import (
        load_benign_clauses,
        run_classifier_benchmark,
    )

    result = run_classifier_benchmark()

    assert len(result.benign_rows) == len(load_benign_clauses())
    assert len(result.adversarial_rows) == 4
    assert 0.0 <= result.false_positive_rate <= 1.0
    assert 0.0 <= result.detection_rate <= 1.0
