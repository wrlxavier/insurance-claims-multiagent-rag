#!/usr/bin/env python3
"""Measure the consistency node over the synthetic claim set [M4-06].

The unit tests (``tests/unit/infrastructure/graph/test_consistency_checks.py``
and ``test_consistency.py``) cover the node's shape and every deterministic
check with a fake LLM. This script is the on-demand live check: run the real
fast model over all 51 synthetic claims (40 ``claims.jsonl`` + 11
``product_claim_mismatch.jsonl``) and report what the two legs actually flag.

What is measurable here. The claim rows carry no per-signal ground truth -- a
"consistency signal" is a judgement call -- so this reports distributions, not
accuracy:

- **Deterministic false-positive discipline.** The 14 ``compatible`` claims are
  coherent and correctly classified by construction, so a deterministic
  ``attention`` signal on that cohort is very likely a false positive. That rate
  is the headline deterministic metric -- the guard rails must not misfire.
- **Deterministic signal counts by check x cohort.** Date and amount checks are
  expected to fire near-zero: the synthetic narratives carry no absolute dates
  and state an amount ~4% of the time (see ``docs/INTAKE_EXTRACTION.md``). That
  is a property of this corpus, not a bug -- the checks are production guard
  rails.
- **LLM signal distribution by category x cohort.** The one place the synthetic
  set exercises the semantic leg: ``unexpected_vagueness`` should fire more on
  the 13 ``insufficient_information`` claims (which deliberately omit a
  load-bearing fact) than on the 14 ``compatible`` ones.
- **Zero-signal ("clean pass") rate** per cohort and the **LLM-failure count**.

The product/claim mismatch cohort's mismatch is claim-vs-target-document; this
node never sees the target document, and intake classifies those events as
``CASCO`` correctly, so the deterministic product-line check is expected to stay
silent on them. See ``docs/CONSISTENCY_NODE.md``.

Writes ``eval/runs/consistency.{md,json}`` and a per-claim
``eval/runs/consistency_signals.jsonl``.
"""

from __future__ import annotations

import dataclasses
import json
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import cast, get_args

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.runtime import Runtime

from infrastructure.config.llm_client_factory import build_chat_model
from infrastructure.config.settings import get_llm_settings
from infrastructure.evaluation.synthetic_claims_schema import SyntheticClaim
from infrastructure.graph.consistency_checks import DETERMINISTIC_CHECK_NAMES
from infrastructure.graph.context import GraphContext
from infrastructure.graph.nodes.consistency import consistency
from infrastructure.graph.nodes.intake import intake
from infrastructure.graph.schemas import ConsistencyCheckName
from infrastructure.graph.state import (
    ClaimState,
    ConsistencyReport,
    ExtractedEntities,
)
from infrastructure.rag.retrieved_clause import RetrievedClause

CLAIM_PATHS = (
    Path("data/synthetic_claims/claims.jsonl"),
    Path("data/synthetic_claims/product_claim_mismatch.jsonl"),
)
REPORT_MD = Path("eval/runs/consistency.md")
REPORT_JSON = Path("eval/runs/consistency.json")
SIGNALS_JSONL = Path("eval/runs/consistency_signals.jsonl")

_MISMATCH_FILE = "product_claim_mismatch.jsonl"
_LLM_CHECK_NAMES: tuple[str, ...] = get_args(ConsistencyCheckName)


class _NoopRetriever:
    """A ``RetrievalPort`` neither node calls."""

    def retrieve(
        self, question: str, *, k: int, metadata_filter: object | None = None
    ) -> list[RetrievedClause]:
        return []


@dataclass
class _ClaimResult:
    claim_id: str
    cohort: str
    narrative: str
    deterministic_signals: list[dict[str, str]] = field(default_factory=list)
    llm_signals: list[dict[str, str]] = field(default_factory=list)
    llm_failed: bool = False
    latency_seconds: float = 0.0
    error: str | None = None

    @property
    def deterministic_attention(self) -> int:
        rows = self.deterministic_signals
        return sum(1 for s in rows if s["severity"] == "attention")

    @property
    def total_signals(self) -> int:
        return len(self.deterministic_signals) + len(self.llm_signals)


@dataclass
class _CohortMetrics:
    n: int
    deterministic_attention_claims: int
    zero_signal_claims: int
    deterministic_by_check: dict[str, int]
    llm_by_check: dict[str, int]


@dataclass
class _Summary:
    processed: int
    error_claim_ids: list[str]
    llm_failed_claim_ids: list[str]
    by_cohort: dict[str, _CohortMetrics]
    compatible_deterministic_false_positive_rate: float
    total_latency_seconds: float


@dataclass
class ConsistencyEvalResult:
    """The aggregate metrics plus every per-claim result row."""

    meta: dict[str, object]
    summary: _Summary
    results: list[_ClaimResult]

    def to_json(self) -> dict[str, object]:
        """Return the JSON-serialisable ``{meta, summary}`` report body."""
        return {"meta": self.meta, "summary": _summary_to_json(self.summary)}


def _summary_to_json(summary: _Summary) -> dict[str, object]:
    body = dataclasses.asdict(summary)
    body["by_cohort"] = {
        cohort: dataclasses.asdict(metrics)
        for cohort, metrics in summary.by_cohort.items()
    }
    return body


def load_claims(
    paths: tuple[Path, ...] = CLAIM_PATHS,
) -> list[tuple[str, SyntheticClaim]]:
    """Return ``(cohort, claim)`` pairs across every claim JSONL."""
    rows: list[tuple[str, SyntheticClaim]] = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            claim = SyntheticClaim.model_validate_json(line)
            cohort = (
                "mismatch"
                if path.name == _MISMATCH_FILE
                else claim.expected_verdict.value
            )
            rows.append((cohort, claim))
    return rows


def _signal_rows(report: ConsistencyReport, source: str) -> list[dict[str, str]]:
    return [
        {"check": s.check, "severity": s.severity, "detail": s.detail}
        for s in report.signals
        if s.source == source
    ]


def _evaluate_claim(
    cohort: str, claim: SyntheticClaim, context: GraphContext
) -> _ClaimResult:
    result = _ClaimResult(
        claim_id=claim.claim_id, cohort=cohort, narrative=claim.narrative
    )
    state: dict[str, object] = {
        "claim_id": claim.claim_id,
        "raw_claim_text": claim.narrative,
    }
    started = time.monotonic()
    try:
        after_intake = intake(cast(ClaimState, state), Runtime(context=context))
        consistency_state: dict[str, object] = {
            **state,
            "entities": cast("ExtractedEntities | None", after_intake["entities"]),
            "missing_information": cast(list[str], after_intake["missing_information"]),
        }
        update = consistency(
            cast(ClaimState, consistency_state), Runtime(context=context)
        )
    except Exception as exc:  # noqa: BLE001 -- recorded, the run continues
        result.error = f"{type(exc).__name__}: {exc}"
        result.latency_seconds = time.monotonic() - started
        return result
    result.latency_seconds = time.monotonic() - started

    report = update["consistency"]
    assert isinstance(report, ConsistencyReport)
    trail = update["audit_trail"]
    assert isinstance(trail, list)

    result.deterministic_signals = _signal_rows(report, "deterministic")
    result.llm_signals = _signal_rows(report, "llm")
    result.llm_failed = "llm_failed=True" in (trail[1].node_input or "")
    return result


def _cohort_metrics(rows: list[_ClaimResult]) -> _CohortMetrics:
    deterministic_by_check = dict.fromkeys(DETERMINISTIC_CHECK_NAMES, 0)
    llm_by_check = dict.fromkeys(_LLM_CHECK_NAMES, 0)
    for row in rows:
        for signal in row.deterministic_signals:
            deterministic_by_check[signal["check"]] = (
                deterministic_by_check.get(signal["check"], 0) + 1
            )
        for signal in row.llm_signals:
            llm_by_check[signal["check"]] = llm_by_check.get(signal["check"], 0) + 1
    return _CohortMetrics(
        n=len(rows),
        deterministic_attention_claims=sum(
            1 for r in rows if r.deterministic_attention > 0
        ),
        zero_signal_claims=sum(1 for r in rows if r.total_signals == 0),
        deterministic_by_check=deterministic_by_check,
        llm_by_check=llm_by_check,
    )


def _compute_summary(
    results: list[_ClaimResult], total_latency_seconds: float
) -> _Summary:
    ok = [r for r in results if r.error is None]
    cohorts = sorted({r.cohort for r in ok})
    by_cohort = {
        cohort: _cohort_metrics([r for r in ok if r.cohort == cohort])
        for cohort in cohorts
    }
    compatible = by_cohort.get("compatible")
    fp_rate = (
        compatible.deterministic_attention_claims / compatible.n
        if compatible and compatible.n
        else 0.0
    )
    return _Summary(
        processed=len(results),
        error_claim_ids=[r.claim_id for r in results if r.error is not None],
        llm_failed_claim_ids=[r.claim_id for r in ok if r.llm_failed],
        by_cohort=by_cohort,
        compatible_deterministic_false_positive_rate=round(fp_rate, 3),
        total_latency_seconds=round(total_latency_seconds, 2),
    )


def run_consistency_eval(
    model: BaseChatModel | None = None,
) -> ConsistencyEvalResult:
    """Run intake + the consistency node over every synthetic claim and aggregate."""
    settings = get_llm_settings()
    fast_model = model or build_chat_model(
        settings,
        settings.llm_model_fast,
        provider_order=settings.llm_fast_provider_order,
        allow_fallbacks=settings.llm_fast_allow_fallbacks,
    )
    context = GraphContext(
        fast_model=fast_model,
        reasoning_model=fast_model,
        retriever=_NoopRetriever(),
        llm_settings=settings,
    )
    claims = load_claims()
    started = time.monotonic()
    results = [_evaluate_claim(cohort, claim, context) for cohort, claim in claims]
    elapsed = time.monotonic() - started

    meta: dict[str, object] = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "model": settings.llm_model_fast,
        "provider_order": settings.llm_fast_provider_order,
        "n_claims": len(claims),
        "note": (
            "No per-signal ground truth -- distributions, not accuracy. Date and "
            "amount deterministic checks are expected near-zero on this corpus "
            "(no absolute dates, amount stated ~4% of the time); they are "
            "production guard rails, unit-tested in isolation. See "
            "docs/CONSISTENCY_NODE.md."
        ),
    }
    return ConsistencyEvalResult(meta, _compute_summary(results, elapsed), results)


def _cohort_table(summary: _Summary) -> list[str]:
    lines = [
        "| cohort | n | claims with a deterministic `attention` | zero-signal claims |",
        "| --- | ---: | ---: | ---: |",
    ]
    for cohort, m in summary.by_cohort.items():
        lines.append(
            f"| {cohort} | {m.n} | {m.deterministic_attention_claims} | "
            f"{m.zero_signal_claims} |"
        )
    return lines


def _by_check_block(summary: _Summary, attr: str, title: str) -> list[str]:
    lines = [
        f"### {title}",
        "",
        "```",
    ]
    payload = {
        cohort: {k: v for k, v in getattr(m, attr).items() if v}
        for cohort, m in summary.by_cohort.items()
    }
    lines.append(json.dumps(payload, indent=2, ensure_ascii=False))
    lines.append("```")
    lines.append("")
    return lines


def render_markdown(result: ConsistencyEvalResult) -> str:
    """Render the ``eval/runs/consistency.md`` body."""
    s = result.summary
    lines = [
        "# Consistency node -- live measurement ([M4-06])",
        "",
        "Generated by `scripts/eval_consistency.py` (`make eval-consistency`). "
        "Regenerable; committed analysis in `docs/CONSISTENCY_NODE.md`.",
        "",
        f"- generated (UTC): `{result.meta['generated_at_utc']}`",
        f"- model: `{result.meta['model']}` "
        f"(provider order `{result.meta['provider_order']}`)",
        f"- processed {s.processed} claims, {len(s.error_claim_ids)} errors "
        f"{s.error_claim_ids or ''}",
        f"- LLM leg failed (degraded to deterministic-only): "
        f"{s.llm_failed_claim_ids or 'none'}",
        f"- wall clock: {s.total_latency_seconds}s",
        f"- note: {result.meta['note']}",
        "",
        "## Deterministic false-positive discipline",
        "",
        "The `compatible` cohort is coherent and correctly classified by "
        "construction, so a deterministic `attention` signal there is very "
        "likely a false positive.",
        "",
        f"- **deterministic `attention` false-positive rate (compatible cohort): "
        f"{s.compatible_deterministic_false_positive_rate:.1%}**",
        "",
        "## Signals by cohort",
        "",
        *_cohort_table(s),
        "",
        *_by_check_block(
            s, "deterministic_by_check", "Deterministic signal counts by check"
        ),
        *_by_check_block(s, "llm_by_check", "LLM signal counts by category"),
        "## Findings",
        "",
        "_To be written from the first run._",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    """Run the measurement and write the reports under ``eval/runs/``."""
    result = run_consistency_eval()

    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(
        json.dumps(result.to_json(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    REPORT_MD.write_text(render_markdown(result), encoding="utf-8")
    with SIGNALS_JSONL.open("w", encoding="utf-8") as handle:
        for row in result.results:
            handle.write(json.dumps(dataclasses.asdict(row), ensure_ascii=False) + "\n")

    print(render_markdown(result))


if __name__ == "__main__":
    main()
