#!/usr/bin/env python3
"""Measure the intake node's extraction over the synthetic claim set [M4-02].

DoD item 5 -- "test against all 30 synthetic claims; report extraction accuracy
per field". Runs the real fast model, so it is run on demand via
``make eval-intake`` and never in CI; the end-to-end verdict accuracy is
[M4-10]'s. The finalized synthetic set has since grown past the DoD's "30" to
40 ``claims.jsonl`` rows + 11 ``product_claim_mismatch.jsonl`` rows; all 51 are
scored and the drift is noted in the report.

What is and is not measurable. The claim rows carry no per-field ground truth
for the free-text entities (``event_type``, ``event_date`` ...), so those are
reported as population rates, not accuracy. The two labels that *are* ground
truth:

- the target document's ``product_line`` (from ``manifest.csv``) -- intake
  should classify the event's line to match it for the 40 in-domain claims. The
  11 mismatch rows describe damage to the insured's own vehicle, so the event
  line is ``CASCO`` regardless of which product the narrative is aimed at.
- ``missing_fact_type`` on the 13 ``insufficient_information`` rows -- intake's
  ``missing_information`` list should contain it.

Writes ``eval/runs/intake_extraction.{md,json}`` and a per-claim
``eval/runs/intake_extraction_predictions.jsonl`` for manual inspection.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import re
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import get_args

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.runtime import Runtime

from infrastructure.config.llm_client_factory import build_chat_model
from infrastructure.config.settings import get_llm_settings
from infrastructure.evaluation.synthetic_claims_schema import SyntheticClaim
from infrastructure.graph.context import GraphContext
from infrastructure.graph.nodes.intake import intake
from infrastructure.graph.schemas import MissingInfoTag
from infrastructure.graph.state import ClaimState, ExtractedEntities
from infrastructure.rag.retrieved_clause import RetrievedClause

CLAIM_PATHS = (
    Path("data/synthetic_claims/claims.jsonl"),
    Path("data/synthetic_claims/product_claim_mismatch.jsonl"),
)
MANIFEST_PATH = Path("data/policies/manifest.csv")
REPORT_MD = Path("eval/runs/intake_extraction.md")
REPORT_JSON = Path("eval/runs/intake_extraction.json")
PREDICTIONS_JSONL = Path("eval/runs/intake_extraction_predictions.jsonl")

_MISMATCH_FILE = "product_claim_mismatch.jsonl"
_SUSEP_PROCESS_RE = re.compile(r"\d{5}\.\d{6}/\d{4}-\d{2}")
_ENTITY_FIELDS: tuple[str, ...] = (
    "event_type",
    "event_date",
    "description",
    "estimated_amount",
    "vehicle_info",
    "susep_process",
    "product_line",
)
_MISSING_TAGS: tuple[str, ...] = get_args(MissingInfoTag)
_VERDICTS: tuple[str, ...] = (
    "compatible",
    "incompatible",
    "insufficient_information",
)


class _NoopRetriever:
    """A ``RetrievalPort`` the intake node never calls."""

    def retrieve(
        self, question: str, *, k: int, metadata_filter: object | None = None
    ) -> list[RetrievedClause]:
        return []


@dataclass
class _Prediction:
    claim_id: str
    source_file: str
    expected_verdict: str
    target_product_line: str
    event_product_line: str  # the line intake's classification is scored against
    missing_fact_type: str | None
    narrative_states_a_process: bool
    predicted_product_line: str | None = None
    predicted_missing: list[str] = field(default_factory=list)
    entities: dict[str, object] = field(default_factory=dict)
    latency_seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    error: str | None = None

    @property
    def invented_process(self) -> bool:
        return (
            self.error is None
            and self.entities.get("susep_process") is not None
            and not self.narrative_states_a_process
        )


@dataclass
class _Count:
    correct: int
    total: int

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0


@dataclass
class _ProductLineMetrics:
    # claims.jsonl, split by expected verdict, scored against the target
    # document's manifest line. The `compatible` cohort is the cleanest signal:
    # the described event belongs to that product line by construction. An
    # `incompatible` claim's event often belongs to a *different* line (that is
    # why it is incompatible), so a "wrong" line there can be the right answer.
    by_verdict_cohort: dict[str, _Count]
    mismatch_as_casco: _Count  # product_claim_mismatch.jsonl, scored against CASCO
    null_predictions: int
    scored: int
    confusion: dict[str, dict[str, int]]


@dataclass
class _MissingInfoMetrics:
    insufficient_n: int
    missing_fact_type_recall: float
    exact_list_match: float
    per_tag_recall: dict[str, _Count]
    answerable_n: int
    false_positive_rate: float
    false_positive_tags: dict[str, int]


@dataclass
class _Summary:
    processed: int
    error_claim_ids: list[str]
    product_line: _ProductLineMetrics
    missing_info: _MissingInfoMetrics
    field_non_null_rate: dict[str, float]
    field_non_null_rate_by_verdict: dict[str, dict[str, float]]
    event_date_null_when_omitted: _Count
    estimated_amount_null_when_omitted: _Count
    invented_process_claim_ids: list[str]
    total_latency_seconds: float
    total_input_tokens: int
    total_output_tokens: int


@dataclass
class IntakeEvalResult:
    """The aggregate metrics plus every per-claim prediction row."""

    meta: dict[str, object]
    summary: _Summary
    predictions: list[_Prediction]

    def to_json(self) -> dict[str, object]:
        """Return the JSON-serialisable ``{meta, summary}`` report body."""
        return {"meta": self.meta, "summary": dataclasses.asdict(self.summary)}


def load_claims(
    paths: tuple[Path, ...] = CLAIM_PATHS,
) -> list[tuple[str, SyntheticClaim]]:
    """Return ``(source_file, claim)`` pairs across every claim JSONL."""
    rows: list[tuple[str, SyntheticClaim]] = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append((path.name, SyntheticClaim.model_validate_json(line)))
    return rows


def _product_line_by_document(manifest_path: Path = MANIFEST_PATH) -> dict[str, str]:
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        return {row["id"]: row["product_line"] for row in csv.DictReader(handle)}


def _predict(
    source_file: str,
    claim: SyntheticClaim,
    document_line: str,
    context: GraphContext,
) -> _Prediction:
    prediction = _Prediction(
        claim_id=claim.claim_id,
        source_file=source_file,
        expected_verdict=claim.expected_verdict.value,
        target_product_line=document_line,
        event_product_line="CASCO" if source_file == _MISMATCH_FILE else document_line,
        missing_fact_type=(
            claim.missing_fact_type.value if claim.missing_fact_type else None
        ),
        narrative_states_a_process=bool(_SUSEP_PROCESS_RE.search(claim.narrative)),
    )
    state: ClaimState = {
        "claim_id": claim.claim_id,
        "raw_claim_text": claim.narrative,
    }
    started = time.monotonic()
    try:
        update = intake(state, Runtime(context=context))
    except Exception as exc:  # noqa: BLE001 -- recorded, the run continues
        prediction.error = f"{type(exc).__name__}: {exc}"
        prediction.latency_seconds = time.monotonic() - started
        return prediction
    prediction.latency_seconds = time.monotonic() - started

    entities = update["entities"]
    assert isinstance(entities, ExtractedEntities)
    missing = update["missing_information"]
    assert isinstance(missing, list)
    trail = update["audit_trail"]
    assert isinstance(trail, list)

    prediction.predicted_product_line = entities.product_line
    prediction.predicted_missing = [str(tag) for tag in missing]
    prediction.entities = entities.model_dump()
    usage = trail[0].token_usage
    if usage is not None:
        prediction.input_tokens = usage.input_tokens
        prediction.output_tokens = usage.output_tokens
    return prediction


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _product_line_metrics(ok: list[_Prediction]) -> _ProductLineMetrics:
    by_cohort = {verdict: _Count(0, 0) for verdict in _VERDICTS}
    mismatch = _Count(0, 0)
    null_predictions = 0
    confusion: dict[str, dict[str, int]] = {}
    for p in ok:
        hit = int(p.predicted_product_line == p.event_product_line)
        if p.source_file == _MISMATCH_FILE:
            mismatch = _Count(mismatch.correct + hit, mismatch.total + 1)
        else:
            cohort = by_cohort[p.expected_verdict]
            by_cohort[p.expected_verdict] = _Count(
                cohort.correct + hit, cohort.total + 1
            )
        null_predictions += int(p.predicted_product_line is None)
        row = confusion.setdefault(p.event_product_line, {})
        key = p.predicted_product_line or "null"
        row[key] = row.get(key, 0) + 1
    return _ProductLineMetrics(
        by_verdict_cohort=by_cohort,
        mismatch_as_casco=mismatch,
        null_predictions=null_predictions,
        scored=len(ok),
        confusion=confusion,
    )


def _missing_info_metrics(ok: list[_Prediction]) -> _MissingInfoMetrics:
    insufficient = [p for p in ok if p.missing_fact_type is not None]
    answerable = [p for p in ok if p.missing_fact_type is None]

    recalled = sum(
        1 for p in insufficient if p.missing_fact_type in p.predicted_missing
    )
    exact = sum(1 for p in insufficient if p.predicted_missing == [p.missing_fact_type])
    per_tag: dict[str, _Count] = {tag: _Count(0, 0) for tag in _MISSING_TAGS}
    for p in insufficient:
        tag = p.missing_fact_type
        if tag in per_tag:
            hit = int(tag in p.predicted_missing)
            per_tag[tag] = _Count(per_tag[tag].correct + hit, per_tag[tag].total + 1)

    fp_tags: dict[str, int] = {}
    flagged = 0
    for p in answerable:
        flagged += int(bool(p.predicted_missing))
        for tag in p.predicted_missing:
            fp_tags[tag] = fp_tags.get(tag, 0) + 1

    return _MissingInfoMetrics(
        insufficient_n=len(insufficient),
        missing_fact_type_recall=_ratio(recalled, len(insufficient)),
        exact_list_match=_ratio(exact, len(insufficient)),
        per_tag_recall=per_tag,
        answerable_n=len(answerable),
        false_positive_rate=_ratio(flagged, len(answerable)),
        false_positive_tags=fp_tags,
    )


def _field_population(
    ok: list[_Prediction],
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    overall: dict[str, float] = {}
    by_verdict: dict[str, dict[str, float]] = {}
    verdict_totals = {
        v: sum(1 for p in ok if p.expected_verdict == v) for v in _VERDICTS
    }
    for name in _ENTITY_FIELDS:
        present = sum(1 for p in ok if p.entities.get(name) is not None)
        overall[name] = _ratio(present, len(ok))
        by_verdict[name] = {
            v: _ratio(
                sum(
                    1
                    for p in ok
                    if p.expected_verdict == v and p.entities.get(name) is not None
                ),
                verdict_totals[v],
            )
            for v in _VERDICTS
        }
    return overall, by_verdict


def _null_when_omitted(ok: list[_Prediction], fact: str, entity_field: str) -> _Count:
    rows = [p for p in ok if p.missing_fact_type == fact]
    null = sum(1 for p in rows if p.entities.get(entity_field) is None)
    return _Count(null, len(rows))


def _compute_summary(predictions: list[_Prediction]) -> _Summary:
    ok = [p for p in predictions if p.error is None]
    overall, by_verdict = _field_population(ok)
    return _Summary(
        processed=len(predictions),
        error_claim_ids=[p.claim_id for p in predictions if p.error is not None],
        product_line=_product_line_metrics(ok),
        missing_info=_missing_info_metrics(ok),
        field_non_null_rate=overall,
        field_non_null_rate_by_verdict=by_verdict,
        event_date_null_when_omitted=_null_when_omitted(
            ok, "data_evento_vigencia", "event_date"
        ),
        estimated_amount_null_when_omitted=_null_when_omitted(
            ok, "valor_franquia_limite", "estimated_amount"
        ),
        invented_process_claim_ids=[
            p.claim_id for p in predictions if p.invented_process
        ],
        total_latency_seconds=round(sum(p.latency_seconds for p in predictions), 2),
        total_input_tokens=sum(p.input_tokens for p in predictions),
        total_output_tokens=sum(p.output_tokens for p in predictions),
    )


def run_intake_eval(model: BaseChatModel | None = None) -> IntakeEvalResult:
    """Run the intake node over every synthetic claim and aggregate the metrics."""
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
    document_line = _product_line_by_document()
    claims = load_claims()
    predictions = [
        _predict(source_file, claim, document_line[claim.document_id], context)
        for source_file, claim in claims
    ]

    counts = {
        "claims.jsonl": sum(1 for name, _ in claims if name == "claims.jsonl"),
        "product_claim_mismatch.jsonl": sum(
            1 for name, _ in claims if name == _MISMATCH_FILE
        ),
        "total": len(claims),
    }
    meta: dict[str, object] = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "model": settings.llm_model_fast,
        "provider_order": settings.llm_fast_provider_order,
        "claim_counts": counts,
        "dod_note": (
            f"DoD [M4-02] item 5 says 30 synthetic claims; the finalized set is "
            f"{counts['claims.jsonl']} + {counts['product_claim_mismatch.jsonl']} "
            f"= {counts['total']}, all scored."
        ),
    }
    return IntakeEvalResult(meta, _compute_summary(predictions), predictions)


def render_markdown(result: IntakeEvalResult) -> str:
    """Render the human-readable ``eval/runs/intake_extraction.md`` body."""
    s = result.summary
    pl = s.product_line
    mi = s.missing_info
    per_tag = {tag: f"{c.correct}/{c.total}" for tag, c in mi.per_tag_recall.items()}
    lines = [
        "# Intake node -- extraction measurement ([M4-02])",
        "",
        f"- generated: `{result.meta['generated_at_utc']}`",
        f"- model: `{result.meta['model']}` "
        f"(provider order `{result.meta['provider_order']}`)",
        f"- claims: {result.meta['claim_counts']}",
        f"- {result.meta['dod_note']}",
        f"- processed {s.processed} claims, {len(s.error_claim_ids)} errors "
        f"{s.error_claim_ids or ''}",
        "",
        "## Product-line classification (event line vs the target document's line)",
        "",
        *(
            f"- {verdict} claims: **{c.accuracy:.1%}** ({c.correct}/{c.total})"
            for verdict, c in pl.by_verdict_cohort.items()
        ),
        f"- product/claim mismatch, classified as CASCO: "
        f"**{pl.mismatch_as_casco.accuracy:.1%}** "
        f"({pl.mismatch_as_casco.correct}/{pl.mismatch_as_casco.total})",
        f"- null classifications: {pl.null_predictions}/{pl.scored}",
        "",
        "The `compatible` cohort is the cleanest signal (event belongs to the "
        "target line by construction). `incompatible` claims often describe an "
        "event from another line -- that is why they are incompatible -- so a "
        "'wrong' line there can be the correct read.",
        "",
        "confusion (expected -> predicted):",
        "",
        "```",
        json.dumps(pl.confusion, indent=2, ensure_ascii=False),
        "```",
        "",
        "## `missing_information`",
        "",
        f"- insufficient_information subset (n={mi.insufficient_n}): "
        f"missing_fact_type recall **{mi.missing_fact_type_recall:.1%}**, "
        f"exact-list match {mi.exact_list_match:.1%}",
        f"- answerable subset (n={mi.answerable_n}): false-positive rate "
        f"**{mi.false_positive_rate:.1%}**, tags {mi.false_positive_tags or '{}'}",
        f"- per-tag recall: {per_tag}",
        "",
        "## Field population (no per-field ground truth -- non-null rates only)",
        "",
        "```",
        json.dumps(
            {
                "overall": {k: round(v, 3) for k, v in s.field_non_null_rate.items()},
                "by_expected_verdict": {
                    k: {vk: round(vv, 3) for vk, vv in v.items()}
                    for k, v in s.field_non_null_rate_by_verdict.items()
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        "```",
        "",
        f"- `event_date` null when `data_evento_vigencia` omitted: "
        f"{s.event_date_null_when_omitted.correct}/{s.event_date_null_when_omitted.total}",
        f"- `estimated_amount` null when `valor_franquia_limite` omitted: "
        f"{s.estimated_amount_null_when_omitted.correct}"
        f"/{s.estimated_amount_null_when_omitted.total}",
        "",
        "## Invented policy identifiers (DoD item 4)",
        "",
        f"- claims where `susep_process` was populated with no matching number in "
        f"the narrative: {s.invented_process_claim_ids or 'none'}",
        "",
        f"## Cost -- input {s.total_input_tokens} / output {s.total_output_tokens} "
        f"tokens, {s.total_latency_seconds}s wall clock",
        "",
    ]
    return "\n".join(lines) + "\n"


def _prediction_row(p: _Prediction) -> dict[str, object]:
    row = dataclasses.asdict(p)
    row["invented_process"] = p.invented_process
    row["latency_seconds"] = round(p.latency_seconds, 2)
    return row


def main() -> None:
    """Run the measurement and write the reports under ``eval/runs/``."""
    result = run_intake_eval()

    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(
        json.dumps(result.to_json(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    REPORT_MD.write_text(render_markdown(result), encoding="utf-8")
    with PREDICTIONS_JSONL.open("w", encoding="utf-8") as handle:
        for prediction in result.predictions:
            row = json.dumps(_prediction_row(prediction), ensure_ascii=False)
            handle.write(row + "\n")

    print(render_markdown(result))


if __name__ == "__main__":
    main()
