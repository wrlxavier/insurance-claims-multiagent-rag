"""Three-class verdict scoring: confusion matrix and per-class rates [M4-10].

Pure functions, no I/O -- the sibling of
[infrastructure.evaluation.retrieval_metrics] for the other half of the
project's ground truth. Retrieval is scored against a set of clause ids; an
assessment is scored against one of [domain.verdict.Verdict]'s three values,
and the interesting number is never the headline accuracy alone but *which*
class the wrong answers moved to. Hence a confusion matrix, always, with
per-class precision and recall beside it.

Extracted from ``scripts/eval_compatibility.py``, which computed this inline
for the golden set before [M4-10] needed the same arithmetic over the
synthetic claims. Both scripts import from here, so the two published
confusion matrices cannot drift apart in definition.

A prediction of ``None`` (the run errored on that item) is **excluded** from
every metric rather than counted as wrong: an item the graph never scored is
not evidence about the graph's accuracy, and the caller reports the exclusions
separately.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from domain.verdict import Verdict

# Column order for every confusion matrix the project publishes. Derived from
# the enum rather than restated, so a fourth verdict could never be added to
# `Verdict` and silently missed by a report.
VERDICTS: tuple[str, ...] = tuple(verdict.value for verdict in Verdict)


@dataclass(frozen=True)
class VerdictMetrics:
    """One scored population: the matrix, the accuracy, and the per-class rates.

    ``confusion`` is ``confusion[expected][predicted] -> count``; ``per_class``
    maps each verdict to ``{"support", "precision", "recall"}``.
    """

    n: int
    accuracy: float
    per_class: dict[str, dict[str, float]]
    confusion: dict[str, dict[str, int]]


def verdict_metrics(pairs: Sequence[tuple[str, str | None]]) -> VerdictMetrics:
    """Score ``(expected, predicted)`` pairs; ``predicted is None`` is skipped.

    Raises on a verdict outside [domain.verdict.Verdict] -- a typo in a label
    or a prediction is a bug in the caller, not a row to score as wrong.
    """
    scored = [(expected, p) for expected, p in pairs if p is not None]
    for expected, predicted in scored:
        for value in (expected, predicted):
            if value not in VERDICTS:
                raise ValueError(f"{value!r} is not one of {VERDICTS}")

    confusion: dict[str, dict[str, int]] = {
        expected: dict.fromkeys(VERDICTS, 0) for expected in VERDICTS
    }
    for expected, predicted in scored:
        confusion[expected][predicted] += 1

    per_class: dict[str, dict[str, float]] = {}
    for verdict in VERDICTS:
        tp = confusion[verdict][verdict]
        fp = sum(confusion[other][verdict] for other in VERDICTS if other != verdict)
        fn = sum(confusion[verdict][other] for other in VERDICTS if other != verdict)
        support = tp + fn
        per_class[verdict] = {
            "support": float(support),
            "precision": tp / (tp + fp) if (tp + fp) else 0.0,
            "recall": tp / support if support else 0.0,
        }

    correct = sum(1 for expected, predicted in scored if expected == predicted)
    return VerdictMetrics(
        n=len(scored),
        accuracy=correct / len(scored) if scored else 0.0,
        per_class=per_class,
        confusion=confusion,
    )


def metrics_json(metrics: VerdictMetrics) -> dict[str, Any]:
    """The JSON-serialisable view every eval report embeds."""
    return {
        "n": metrics.n,
        "accuracy": metrics.accuracy,
        "per_class": metrics.per_class,
        "confusion": metrics.confusion,
    }


def confusion_table_lines(metrics: VerdictMetrics) -> list[str]:
    """Render the matrix as Markdown table rows (header + divider + one row each)."""
    header = "| expected \\ predicted | " + " | ".join(VERDICTS) + " |"
    divider = "| --- | " + " | ".join("---:" for _ in VERDICTS) + " |"
    lines = [header, divider]
    for expected in VERDICTS:
        cells = " | ".join(str(metrics.confusion[expected][p]) for p in VERDICTS)
        lines.append(f"| {expected} | {cells} |")
    return lines


def per_class_table_lines(metrics: VerdictMetrics) -> list[str]:
    """Render precision/recall/support as Markdown table rows."""
    lines = [
        "| verdict | support | precision | recall |",
        "| --- | ---: | ---: | ---: |",
    ]
    for verdict in VERDICTS:
        stats = metrics.per_class[verdict]
        lines.append(
            f"| {verdict} | {int(stats['support'])} | "
            f"{stats['precision']:.1%} | {stats['recall']:.1%} |"
        )
    return lines
