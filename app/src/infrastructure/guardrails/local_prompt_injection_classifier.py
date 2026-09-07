"""The real, local ``InjectionClassifierPort`` -- [M5-08 Appendix].

Loads the pinned ``protectai/deberta-v3-base-prompt-injection-v2`` (see
``infrastructure.guardrails.classifier_config``) with a ``transformers``
``text-classification`` pipeline and scores text in-process: no API, no rate
limit, no per-token charge.

``transformers`` (and its ``torch`` dependency) is the optional ``embed``
dependency group -- the same one
``infrastructure.rag.sentence_transformer_embedder``/
``infrastructure.rag.cross_encoder_reranker`` use, deliberately kept out of
``uv sync`` and CI. Import this module freely -- the heavy import is
deferred to ``LocalPromptInjectionClassifier.__init__``; only constructing
one needs the group installed (``make eval-prompt-injection-classifier``
runs ``uv run --group embed``).
"""

from __future__ import annotations

import logging

from infrastructure.graph.context import ClassificationResult
from infrastructure.guardrails.classifier_config import (
    CLASSIFIER_MAX_INPUT_TOKENS,
    CLASSIFIER_MODEL_REVISION,
    CLASSIFIER_POSITIVE_LABEL,
)

_logger = logging.getLogger(__name__)


class LocalPromptInjectionClassifier:
    """Score text for injected-instruction risk with the pinned model, loaded once."""

    def __init__(
        self,
        *,
        model_id: str,
        threshold: float,
        device: str | None = None,
    ) -> None:
        """Load the pinned model at its pinned revision.

        ``model_id`` is the caller's (settings-sourced) human-facing name --
        see ``classifier_config``'s module docstring for why it is
        cross-checked against ``CLASSIFIER_MODEL_ID`` rather than hardcoded
        here. ``device`` is passed straight to the pipeline -- ``None``
        auto-selects (GPU if available, else CPU); pass ``"cpu"`` to force it.
        """
        try:
            from transformers import pipeline
        except ModuleNotFoundError as exc:  # pragma: no cover - env-dependent
            raise ModuleNotFoundError(
                "transformers is not installed. It is the optional `embed` "
                "dependency group: run `make eval-prompt-injection-classifier` "
                "(which uses `uv run --group embed`) or `uv sync --group embed` "
                "first."
            ) from exc

        self._threshold = threshold
        self._pipeline = pipeline(
            "text-classification",
            model=model_id,
            revision=CLASSIFIER_MODEL_REVISION,
            top_k=None,
            truncation=True,
            max_length=CLASSIFIER_MAX_INPUT_TOKENS,
            device=device,
        )

    def classify(self, text: str, *, source: str) -> ClassificationResult:
        """Score ``text`` (read from ``source``) for injected-instruction risk.

        Never raises -- ``InjectionClassifierPort``'s contract. A pipeline
        failure (a malformed input, an OOM, anything) is logged and reported
        as unflagged rather than propagated: this guardrail is advisory, so
        the run it would otherwise interrupt is worth more than the signal.
        """
        try:
            predictions = self._pipeline(text)[0]
            scores = {row["label"]: float(row["score"]) for row in predictions}
            top_label = max(predictions, key=lambda row: row["score"])["label"]
        except Exception:
            _logger.exception(
                "prompt_injection_classifier.failed", extra={"source": source}
            )
            return ClassificationResult(flagged=False, score=0.0, label="error")

        score = scores.get(CLASSIFIER_POSITIVE_LABEL, 0.0)
        return ClassificationResult(
            flagged=score >= self._threshold, score=score, label=top_label
        )
