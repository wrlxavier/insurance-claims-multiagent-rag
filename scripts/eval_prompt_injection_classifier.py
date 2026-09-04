#!/usr/bin/env python3
"""Domain benchmark for the optional runtime classifier ([M5-08 Appendix]).

The Appendix's own ask, verbatim: "Measure false-positive rates on
imperative SUSEP clauses ... and evaluate per-node latency." This script
loads the real
``infrastructure.guardrails.local_prompt_injection_classifier.LocalPromptInjectionClassifier``
(needs the optional ``embed`` uv group -- no LLM, no Postgres) and scores two
sets:

- ``data/adversarial_injection/benign_imperative_clauses.jsonl`` -- ten real,
  non-adversarial imperative SUSEP clause excerpts. A flagged row is a false
  positive.
- ``data/adversarial_injection/fixtures.jsonl`` (reusing
  ``scripts.eval_prompt_injection.load_fixtures``) -- the four hand-authored
  adversarial probes' injected spans (a poisoned clause excerpt, or the
  injected half of a clean/injected claim-narrative pair). An unflagged row
  is a missed detection.

No pass/fail gate: this is a measurement script, not a proof that the
containment defenses hold (``scripts/eval_prompt_injection.py`` is that
proof, and stays the thing this project actually relies on). The numbers
here drive a written adopt/keep-off recommendation in
``docs/PROMPT_INJECTION_CLASSIFIER.md`` for
``PROMPT_INJECTION_CLASSIFIER_ENABLED``'s default.

Run via ``make eval-prompt-injection-classifier``. Writes
``eval/runs/prompt_injection_classifier.{md,json}``.
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from infrastructure.config.settings import get_llm_settings
from infrastructure.evaluation.benign_clause_schema import BenignClauseFixture
from infrastructure.guardrails.local_prompt_injection_classifier import (
    LocalPromptInjectionClassifier,
)
from scripts.eval_prompt_injection import FIXTURES_PATH, load_fixtures
from scripts.tune_reranking import summarise_latency

BENIGN_PATH = Path("data/adversarial_injection/benign_imperative_clauses.jsonl")
OUTPUT_DIR = Path("eval/runs")
JSON_PATH = OUTPUT_DIR / "prompt_injection_classifier.json"
MD_PATH = OUTPUT_DIR / "prompt_injection_classifier.md"

SCHEMA_VERSION = "v1"


def load_benign_clauses(path: Path = BENIGN_PATH) -> list[BenignClauseFixture]:
    """Load and validate every benign-clause row; a malformed row fails loudly."""
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(BenignClauseFixture.model_validate_json(line))
            except Exception as exc:  # noqa: BLE001 - re-raised with location context
                raise ValueError(f"{path}:{line_no} is not a valid row: {exc}") from exc
    return rows


def _adversarial_spans() -> list[tuple[str, str]]:
    """One (id, text) pair per injected span across the four M5-08 fixtures.

    ``clause_injection`` contributes its poisoned ``excerpt``;
    ``claim_injection`` contributes ``claim_narrative_injected``. Every span
    here is text the M5-08 eval already proved does not hijack a verdict --
    this script asks the separate question of whether the classifier notices
    it at all.
    """
    spans = []
    for fixture in load_fixtures(FIXTURES_PATH):
        if fixture.kind == "clause_injection":
            assert fixture.citations
            spans.append((fixture.fixture_id, fixture.citations[0].excerpt))
        else:
            assert fixture.claim_narrative_injected is not None
            spans.append((fixture.fixture_id, fixture.claim_narrative_injected))
    return spans


@dataclass(frozen=True)
class ScoreRow:
    """One span's classification, timed."""

    span_id: str
    expected_label: str  # "benign" or "adversarial"
    flagged: bool
    score: float
    label: str
    latency_ms: float

    @property
    def correct(self) -> bool:
        """Whether ``flagged`` matches what ``expected_label`` predicts."""
        return self.flagged == (self.expected_label == "adversarial")


@dataclass(frozen=True)
class ClassifierBenchmarkResult:
    """Everything ``make eval-prompt-injection-classifier`` produces."""

    meta: dict[str, Any]
    benign_rows: list[ScoreRow]
    adversarial_rows: list[ScoreRow]

    @property
    def false_positive_rate(self) -> float:
        """Fraction of the real, non-adversarial clauses the classifier flagged."""
        if not self.benign_rows:
            return 0.0
        return sum(1 for r in self.benign_rows if r.flagged) / len(self.benign_rows)

    @property
    def detection_rate(self) -> float:
        """Fraction of the adversarial spans the classifier flagged."""
        if not self.adversarial_rows:
            return 0.0
        return sum(1 for r in self.adversarial_rows if r.flagged) / len(
            self.adversarial_rows
        )

    def to_json(self) -> dict[str, Any]:
        """The JSON-serialisable view this script writes to disk."""
        all_rows = self.benign_rows + self.adversarial_rows
        return {
            "schema_version": SCHEMA_VERSION,
            "meta": self.meta,
            "false_positive_rate": self.false_positive_rate,
            "detection_rate": self.detection_rate,
            "latency_ms": summarise_latency([r.latency_ms for r in all_rows]),
            "rows": [
                {
                    "span_id": r.span_id,
                    "expected_label": r.expected_label,
                    "flagged": r.flagged,
                    "score": r.score,
                    "label": r.label,
                    "latency_ms": round(r.latency_ms, 2),
                    "correct": r.correct,
                }
                for r in all_rows
            ],
        }


def run_classifier_benchmark() -> ClassifierBenchmarkResult:
    """Score the benign and adversarial sets with the real local classifier."""
    settings = get_llm_settings()
    classifier = LocalPromptInjectionClassifier(
        model_id=settings.prompt_injection_classifier_model,
        threshold=settings.prompt_injection_classifier_threshold,
        device="cpu",
    )

    benign_rows: list[ScoreRow] = []
    for row in load_benign_clauses():
        started = time.perf_counter()
        result = classifier.classify(row.excerpt, source=row.clause_id)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        benign_rows.append(
            ScoreRow(
                span_id=row.fixture_id,
                expected_label="benign",
                flagged=result.flagged,
                score=result.score,
                label=result.label,
                latency_ms=elapsed_ms,
            )
        )
        print(
            f"{row.fixture_id:<40} benign      "
            f"score={result.score:.3f} {'FLAGGED' if result.flagged else 'ok'}",
            flush=True,
        )

    adversarial_rows: list[ScoreRow] = []
    for span_id, text in _adversarial_spans():
        started = time.perf_counter()
        result = classifier.classify(text, source=span_id)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        adversarial_rows.append(
            ScoreRow(
                span_id=span_id,
                expected_label="adversarial",
                flagged=result.flagged,
                score=result.score,
                label=result.label,
                latency_ms=elapsed_ms,
            )
        )
        print(
            f"{span_id:<40} adversarial "
            f"score={result.score:.3f} {'FLAGGED' if result.flagged else 'MISSED'}",
            flush=True,
        )

    meta = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "model": settings.prompt_injection_classifier_model,
        "threshold": settings.prompt_injection_classifier_threshold,
        "benign_path": str(BENIGN_PATH),
        "fixtures_path": str(FIXTURES_PATH),
        "benign_count": len(benign_rows),
        "adversarial_count": len(adversarial_rows),
        "platform": platform.platform(),
    }
    return ClassifierBenchmarkResult(
        meta=meta, benign_rows=benign_rows, adversarial_rows=adversarial_rows
    )


def render_markdown(result: ClassifierBenchmarkResult) -> str:
    """Render the run as Markdown; the numbers are copied into the doc."""
    lines = [
        "# Prompt-injection classifier -- domain benchmark ([M5-08 Appendix])",
        "",
        "Generated by `scripts/eval_prompt_injection_classifier.py` "
        "(`make eval-prompt-injection-classifier`): the real "
        "`infrastructure.guardrails.local_prompt_injection_classifier."
        "LocalPromptInjectionClassifier` "
        f"(`{result.meta['model']}`, threshold `{result.meta['threshold']}`), over "
        f"{result.meta['benign_count']} real benign clause excerpts "
        f"(`{result.meta['benign_path']}`) and "
        f"{result.meta['adversarial_count']} adversarial spans from "
        f"`{result.meta['fixtures_path']}`. Committed analysis: "
        "`docs/PROMPT_INJECTION_CLASSIFIER.md`.",
        "",
        f"- Generated (UTC): {result.meta['generated_at_utc']}",
        f"- Platform: {result.meta['platform']}",
        f"- **False-positive rate (benign clauses flagged): "
        f"{result.false_positive_rate:.1%}**",
        f"- **Detection rate (adversarial spans flagged): "
        f"{result.detection_rate:.1%}**",
        "",
        "| span | expected | flagged | score | label | latency (ms) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in result.benign_rows + result.adversarial_rows:
        lines.append(
            f"| {r.span_id} | {r.expected_label} | "
            f"{'✅' if r.flagged else '—'} | {r.score:.3f} | {r.label} | "
            f"{r.latency_ms:.1f} |"
        )
    latency = summarise_latency(
        [r.latency_ms for r in result.benign_rows + result.adversarial_rows]
    )
    lines += [
        "",
        f"Latency: n={latency['n']}, p50={latency['p50']}ms, "
        f"p95={latency['p95']}ms, mean={latency['mean']}ms.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    """Run the benchmark and write the eval/runs report -- json and markdown."""
    result = run_classifier_benchmark()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(
        json.dumps(result.to_json(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    MD_PATH.write_text(render_markdown(result), encoding="utf-8")
    print("")
    print(f"false-positive rate: {result.false_positive_rate:.1%}")
    print(f"detection rate: {result.detection_rate:.1%}")
    print(f"Wrote {JSON_PATH} and {MD_PATH}")


if __name__ == "__main__":
    main()
