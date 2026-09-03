#!/usr/bin/env python3
"""End-to-end verdict accuracy and citation coverage over the synthetic claims [M4-10].

Every other M4 eval measures one node. This one compiles the **whole graph**
(``build_claim_graph()``) and runs a claim narrative through it -- intake, the
clarification loop, retrieval, the parallel compatibility/consistency branches,
the recommendation node and the [M4-09] human checkpoint -- then asks whether
the verdict at the far end matches the claim's label. It is the number M4's
exit criteria name, and the only one that catches a failure living *between*
two nodes rather than inside one.

Reported: the 3x3 verdict confusion matrix with per-class precision/recall,
overall and per cohort (the product/claim-mismatch subset separately, as its
own cohort); the graph completion rate; a failure catalogue attributing every
wrong verdict to a cause; reference-clause recall; and -- with ``--judge`` --
faithfulness and context relevance from the independent judge in
[infrastructure.evaluation.judge], three passes per item with the variance
published beside the mean.

**The claim carries its policy.** A real claim is filed *against* a policy, but
the synthetic narratives never state one and ``ClaimState`` has no field for
it, so by default this script prepends a one-line policy reference (the target
document's SUSEP process, from the manifest) to the narrative and lets intake
extract it the way it would from any claim that mentions its policy. Without
it the product/claim-mismatch cohort is unanswerable by construction: the graph
would never learn which product the claimant actually bought.
``--no-policy-header`` runs the other arm and measures what that costs -- the
"claims path where a process is not stated" ``docs/RETRIEVAL_NODE.md`` left to
this issue.

**The checkpoint is resumed, not skipped.** ``human_review`` interrupts
unconditionally; this script supplies an ``InMemorySaver`` and resumes with a
canned ``approve``, as ``docs/HUMAN_CHECKPOINT.md`` says [M4-10] should. The
auto-approval is a mechanical resume, not a human judgment: every number here
scores the **system's** recommendation, never a reviewed one.

Needs a running Postgres with loaded + embedded chunks, the optional ``embed``
uv group, and ``LLM_*`` in ``.env``. Run via ``make eval-end-to-end`` (the
headline arm). Writes ``eval/runs/<stem>.{md,json}`` and a per-claim
``eval/runs/<stem>_predictions.jsonl``; with ``--write-snapshot`` it also
writes the **committed** ``eval/end_to_end_citations.json`` that
``make validate-citation-coverage`` replays in CI. The committed analysis lives
in ``docs/END_TO_END_EVALUATION.md``.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from random import Random
from typing import Any, cast

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from infrastructure.config.llm_client_factory import build_chat_model
from infrastructure.config.settings import get_llm_settings
from infrastructure.database import (
    assert_chunk_table_ready,
    create_engine_from_settings,
    create_session_factory,
)
from infrastructure.evaluation.judge import (
    JUDGE_MODEL,
    JUDGE_PASSES,
    JudgeAggregate,
    build_judge_model,
    judge_context_relevance,
    judge_faithfulness,
)
from infrastructure.evaluation.synthetic_claims_schema import SyntheticClaim
from infrastructure.evaluation.verdict_metrics import (
    VerdictMetrics,
    confusion_table_lines,
    metrics_json,
    per_class_table_lines,
    verdict_metrics,
)
from infrastructure.graph.build import build_claim_graph
from infrastructure.graph.checkpointer import build_checkpoint_serializer
from infrastructure.graph.context import GraphContext
from infrastructure.graph.reasoning_format import parse_reasoning
from infrastructure.graph.state import AuditEvent, Citation, Recommendation
from infrastructure.parsing.corpus_artifact import JSONL_PATH
from scripts.eval_consistency import load_claims
from scripts.eval_retrieval import (
    MANIFEST_PATH,
    load_chunk_corpus,
    load_corpus,
    load_document_metadata,
)
from scripts.eval_retrieval_node import _build_adapter

SCHEMA_VERSION = "v1"
OUTPUT_DIR = Path("eval/runs")
DEFAULT_STEM = "end_to_end"
# Committed (eval/runs/ is gitignored, eval/*.json is not) -- the artefact the
# CI citation gate replays offline. See scripts/validate_citation_coverage.py.
SNAPSHOT_PATH = Path("eval/end_to_end_citations.json")

# The decision the checkpoint is resumed with. Approving is the neutral choice:
# it records no opinion of its own, so the recommendation this script scores is
# exactly what the graph produced.
_RESUME_DECISION: dict[str, Any] = {
    "decision": "approve",
    "notes": "aprovação automática do harness [M4-10]; nenhuma revisão humana ocorreu",
}

# `Recommendation` carries no verdict field by [M4-08]'s design -- everything
# load-bearing is derived, and the derivation is recorded in the node's audit
# event as `posture=... verdict=...`. The effective verdict is read from there
# directly; the posture map below is the fallback, and says why an abstention
# happened as well as that it did.
_POSTURE_RE = re.compile(r"posture=(\S+)")
_VERDICT_RE = re.compile(r"verdict=(\S+)")
_POSTURE_VERDICT: dict[str, str] = {
    "compatible": "compatible",
    "incompatible": "incompatible",
    "inconclusive": "insufficient_information",
    "claimant_gaps": "insufficient_information",
    "retrieval_miss": "insufficient_information",
    "no_assessment": "insufficient_information",
}

# The marker the compatibility node writes when three grounding attempts all
# failed and it degraded to insufficient_information.
_DEGRADED_MARKER = "ungrounded_after"
_CLAUSE_ID_RE = re.compile(r"[A-Za-z0-9][\w./:-]*:[\w.-]+")

# Ceiling on one claim's whole-graph run. Higher than the per-node evals' 180s:
# a claim here can pay for two clarification rounds (two intake calls plus two
# clarification calls) before retrieval even starts, then compatibility with up
# to three grounding attempts, then consistency and the justification.
_CLAIM_TIMEOUT_SECONDS = 420.0

# Seed for the claim shuffle, so ``--limit`` samples across cohorts (the claim
# files are grouped by verdict) and a smoke run is still representative.
_SHUFFLE_SEED = 0

_FAILURE_CAUSES: tuple[str, ...] = (
    "claimant_gaps",
    "retrieval_miss",
    "parsing_error",
    "reasoning_error",
)


@dataclass(frozen=True)
class _ClaimResult:
    """One claim's whole-graph run, scored."""

    claim_id: str
    cohort: str
    document_id: str
    expected_verdict: str
    predicted_verdict: str | None
    posture: str | None
    confidence: float | None
    reached_checkpoint: bool
    completed: bool
    context_sufficient: bool | None
    clarification_rounds: int
    clarification_exhausted: bool
    compatibility_verdict: str | None
    grounding_degraded: bool
    justification_degraded: bool
    justification_names_a_clause: bool
    assertions: tuple[tuple[str, tuple[str, ...]], ...]
    retrieved_clause_ids: tuple[str, ...]
    recommendation_citation_ids: tuple[str, ...]
    reference_clause_ids: tuple[str, ...]
    n_consistency_flags: int
    latency_seconds: float
    error: str | None = None

    @property
    def correct(self) -> bool:
        return self.predicted_verdict == self.expected_verdict

    @property
    def reference_hits(self) -> int:
        return len(set(self.reference_clause_ids) & set(self.retrieved_clause_ids))

    @property
    def reference_recall(self) -> float:
        if not self.reference_clause_ids:
            return 0.0
        return self.reference_hits / len(set(self.reference_clause_ids))

    @property
    def retrieved_document_ids(self) -> set[str]:
        """The documents retrieval landed in -- a clause id is ``{document}:{path}``."""
        return {cid.split(":", 1)[0] for cid in self.retrieved_clause_ids if ":" in cid}

    @property
    def on_target_document(self) -> bool:
        """Whether anything retrieved came from the policy the claim was filed against.

        Separates "retrieval found the wrong document" from "retrieval found the
        right document but not the labelled clause" -- two very different
        diagnoses that ``reference_hits == 0`` alone conflates.
        """
        return self.document_id in self.retrieved_document_ids


@dataclass(frozen=True)
class _JudgeSummary:
    """The judge's aggregate, with the stability of the aggregate beside it."""

    passes: int
    n_items: int
    faithfulness_rate: float
    partially_supported_rate: float
    unsupported_rate: float
    faithfulness_unanimous_rate: float
    context_relevance_rate: float
    n_clauses_judged: int
    context_unanimous_rate: float
    per_pass_faithfulness: tuple[float, ...]
    per_pass_context_relevance: tuple[float, ...]

    @property
    def faithfulness_spread(self) -> float:
        values = self.per_pass_faithfulness
        return max(values) - min(values) if values else 0.0

    @property
    def context_relevance_spread(self) -> float:
        values = self.per_pass_context_relevance
        return max(values) - min(values) if values else 0.0


@dataclass(frozen=True)
class EndToEndEvalResult:
    """Everything ``make eval-end-to-end`` produces, for the report + the test."""

    meta: dict[str, Any]
    overall: VerdictMetrics
    by_cohort: dict[str, VerdictMetrics]
    completion_rate: float
    failure_causes: dict[str, int]
    reference_recall_micro: float
    reference_recall_any_rate: float
    on_target_document_rate: float
    justification_names_a_clause_rate: float
    judge: _JudgeSummary | None
    results: list[_ClaimResult]
    error_claim_ids: list[str] = field(default_factory=list)

    @property
    def n_scored(self) -> int:
        """Claims that produced a verdict -- errored runs are excluded."""
        return self.overall.n

    def to_json(self) -> dict[str, Any]:
        """The JSON-serialisable view written to ``eval/runs/<stem>.json``."""
        return {
            "schema_version": SCHEMA_VERSION,
            "meta": self.meta,
            "overall": metrics_json(self.overall),
            "by_cohort": {name: metrics_json(m) for name, m in self.by_cohort.items()},
            "completion_rate": self.completion_rate,
            "failure_causes": self.failure_causes,
            "reference_recall_micro": self.reference_recall_micro,
            "reference_recall_any_rate": self.reference_recall_any_rate,
            "on_target_document_rate": self.on_target_document_rate,
            "justification_names_a_clause_rate": self.justification_names_a_clause_rate,
            "judge": (
                {
                    **vars(self.judge),
                    "faithfulness_spread": self.judge.faithfulness_spread,
                    "context_relevance_spread": self.judge.context_relevance_spread,
                }
                if self.judge is not None
                else None
            ),
            "error_claim_ids": self.error_claim_ids,
        }


def build_claim_text(
    claim: SyntheticClaim, manifest_row: dict[str, str], *, policy_header: bool
) -> str:
    """The graph's ``raw_claim_text``: the narrative, optionally with its policy.

    The header states the registered policy's SUSEP process the way a claim
    submitted against a known policy would carry it. It adds no information
    about the *event* -- only about which product the claim was filed against,
    which is the one fact a mismatch claim turns on and the narrative alone
    can never supply.
    """
    if not policy_header:
        return claim.narrative
    return (
        f"[Apólice registrada: processo SUSEP {manifest_row['susep_process']}]\n"
        f"{claim.narrative}"
    )


def _posture_of(audit_trail: Sequence[AuditEvent]) -> str | None:
    for event in reversed(list(audit_trail)):
        if event.node == "recommendation" and event.node_input:
            match = _POSTURE_RE.search(event.node_input)
            if match:
                return match.group(1)
    return None


def _effective_verdict(
    audit_trail: Sequence[AuditEvent], posture: str | None
) -> str | None:
    """The end-to-end verdict the recommendation node settled on.

    Read from the node's own audit record rather than re-derived here, so the
    published accuracy scores what the graph decided and not this script's
    reading of it. Falls back to the posture map only if the record is missing.
    """
    for event in reversed(list(audit_trail)):
        if event.node == "recommendation" and event.node_input:
            match = _VERDICT_RE.search(event.node_input)
            if match and match.group(1) in _POSTURE_VERDICT.values():
                return match.group(1)
    return _POSTURE_VERDICT.get(posture or "")


def _grounding_degraded(audit_trail: Sequence[AuditEvent]) -> bool:
    return any(
        event.node == "compatibility"
        and event.node_input is not None
        and _DEGRADED_MARKER in event.node_input
        for event in audit_trail
    )


def _justification_degraded(audit_trail: Sequence[AuditEvent]) -> bool:
    """True when the recommendation node fell back to its deterministic template."""
    for event in reversed(list(audit_trail)):
        if event.node == "recommendation" and event.node_input:
            return "llm_failed=True" in event.node_input
    return False


def _run_graph(
    claim: SyntheticClaim,
    claim_text: str,
    context: GraphContext,
) -> tuple[dict[str, Any], bool]:
    """Invoke the compiled graph to the checkpoint, then resume past it.

    Returns ``(final_state, reached_checkpoint)``. A run that never reaches the
    interrupt is a bug worth seeing in the report, not an exception: the graph
    is supposed to be unable to finish without passing the human checkpoint.
    """
    # InMemorySaver, not the Postgres one: nothing here needs to survive the
    # process, and an eval should not leave rows in the service's checkpoint
    # tables. But the *serializer* is the project's, allowlist and all -- without
    # it LangGraph rebuilds every state model as a plain dict, and the scoring
    # below would read `compatibility.verdict` off a mapping (see
    # docs/HUMAN_CHECKPOINT.md, finding 4).
    saver = InMemorySaver(serde=build_checkpoint_serializer())
    compiled = build_claim_graph().compile(checkpointer=saver)
    config: Any = {"configurable": {"thread_id": f"m4-10-{claim.claim_id}"}}
    paused = compiled.invoke(
        {"claim_id": claim.claim_id, "raw_claim_text": claim_text},
        config=config,
        context=context,
    )
    if "__interrupt__" not in paused:
        return cast(dict[str, Any], paused), False
    final = compiled.invoke(
        Command(resume=_RESUME_DECISION), config=config, context=context
    )
    return cast(dict[str, Any], final), True


def _score_claim(
    claim: SyntheticClaim,
    cohort: str,
    state: dict[str, Any],
    *,
    reached_checkpoint: bool,
    latency_seconds: float,
) -> _ClaimResult:
    audit_trail = cast(list[AuditEvent], state.get("audit_trail") or [])
    posture = _posture_of(audit_trail)
    citations = cast(list[Citation], state.get("citations") or [])
    rec = cast(Recommendation | None, state.get("recommendation"))
    compat = state.get("compatibility")

    reasoning = getattr(compat, "reasoning", "") or ""
    assertions = tuple(
        (a.statement, tuple(a.clause_ids)) for a in parse_reasoning(reasoning)
    )
    justification = rec.justification if rec is not None else ""
    return _ClaimResult(
        claim_id=claim.claim_id,
        cohort=cohort,
        document_id=claim.document_id,
        expected_verdict=claim.expected_verdict.value,
        predicted_verdict=_effective_verdict(audit_trail, posture),
        posture=posture,
        confidence=rec.confidence if rec is not None else None,
        reached_checkpoint=reached_checkpoint,
        completed=rec is not None,
        context_sufficient=cast("bool | None", state.get("context_sufficient")),
        clarification_rounds=int(state.get("clarification_rounds", 0) or 0),
        clarification_exhausted=bool(state.get("clarification_exhausted", False)),
        compatibility_verdict=(compat.verdict.value if compat is not None else None),
        grounding_degraded=_grounding_degraded(audit_trail),
        justification_degraded=_justification_degraded(audit_trail),
        justification_names_a_clause=bool(_CLAUSE_ID_RE.search(justification)),
        assertions=assertions,
        retrieved_clause_ids=tuple(c.clause_id for c in citations),
        recommendation_citation_ids=tuple(
            c.clause_id for c in (rec.citations if rec is not None else [])
        ),
        reference_clause_ids=tuple(claim.reference_clause_ids),
        n_consistency_flags=len(rec.consistency_flags) if rec is not None else 0,
        latency_seconds=latency_seconds,
    )


def failure_cause(row: _ClaimResult) -> str | None:
    """Attribute one wrong verdict to a cause; ``None`` when the verdict is right.

    First match wins, and the order is causal rather than alphabetical: an
    exhausted clarification loop never reaches retrieval, so it must be tested
    before any retrieval signal; a retrieval miss starves the assessment, so it
    must be tested before any assessment signal.

    The DoD names three causes. ``claimant_gaps`` is a fourth, because
    ``docs/ARCHITECTURE.md`` already reserves ``clarification_exhausted`` as
    "a distinct failure mode [M4-10] catalogues" -- a claim the graph could not
    assess because the claimant never supplied a load-bearing fact is not a
    retrieval, parsing or reasoning failure, and folding it into any of the
    three would misattribute it.
    """
    if row.error is not None or row.correct:
        return None
    if row.clarification_exhausted:
        return "claimant_gaps"
    if row.context_sufficient is False or row.reference_hits == 0:
        return "retrieval_miss"
    if row.grounding_degraded:
        return "parsing_error"
    return "reasoning_error"


def _judge_run(
    model: BaseChatModel,
    rows: Sequence[_ClaimResult],
    claim_texts: dict[str, str],
    excerpts: dict[str, dict[str, str]],
    *,
    passes: int,
) -> _JudgeSummary:
    """Judge faithfulness per assertion and relevance per retrieved clause."""
    faith: list[JudgeAggregate] = []
    relevance: list[JudgeAggregate] = []
    for row in rows:
        if row.error is not None:
            continue
        claim_text = claim_texts[row.claim_id]
        clause_text = excerpts[row.claim_id]
        if row.assertions:
            faith.extend(
                judge_faithfulness(
                    model,
                    claim_text,
                    [(s, list(ids)) for s, ids in row.assertions],
                    clause_text,
                    passes=passes,
                )
            )
        if row.retrieved_clause_ids:
            relevance.extend(
                judge_context_relevance(
                    model,
                    claim_text,
                    [
                        (cid, clause_text.get(cid, ""))
                        for cid in row.retrieved_clause_ids
                    ],
                    passes=passes,
                ).values()
            )
        print(
            f"  judged {row.claim_id:<28} assertions={len(row.assertions)} "
            f"clauses={len(row.retrieved_clause_ids)}",
            flush=True,
        )

    counts = Counter(a.majority for a in faith)
    n_faith = len(faith) or 1
    n_rel = len(relevance) or 1
    return _JudgeSummary(
        passes=passes,
        n_items=len(faith),
        faithfulness_rate=counts["supported"] / n_faith,
        partially_supported_rate=counts["partially_supported"] / n_faith,
        unsupported_rate=counts["unsupported"] / n_faith,
        faithfulness_unanimous_rate=(
            sum(1 for a in faith if a.unanimous) / n_faith if faith else 0.0
        ),
        context_relevance_rate=(
            sum(1 for a in relevance if a.majority == "relevant") / n_rel
        ),
        n_clauses_judged=len(relevance),
        context_unanimous_rate=(
            sum(1 for a in relevance if a.unanimous) / n_rel if relevance else 0.0
        ),
        per_pass_faithfulness=_per_pass_rate(faith, "supported"),
        per_pass_context_relevance=_per_pass_rate(relevance, "relevant"),
    )


def _per_pass_rate(
    aggregates: Sequence[JudgeAggregate], positive: str
) -> tuple[float, ...]:
    """The metric recomputed from each pass alone -- the spread is the variance."""
    if not aggregates:
        return ()
    n_passes = aggregates[0].n_passes
    rates = []
    for index in range(n_passes):
        hits = sum(1 for a in aggregates if a.pass_values[index] == positive)
        rates.append(hits / len(aggregates))
    return tuple(rates)


def run_end_to_end_eval(
    *,
    policy_header: bool = True,
    judge: bool = False,
    limit: int | None = None,
    judge_passes: int = JUDGE_PASSES,
) -> EndToEndEvalResult:
    """Run the whole graph over every synthetic claim and score the final verdict."""
    settings = get_llm_settings()
    reasoning_model = build_chat_model(
        settings,
        settings.llm_model_reasoning,
        provider_order=settings.llm_reasoning_provider_order,
        allow_fallbacks=settings.llm_reasoning_allow_fallbacks,
    )
    fast_model = build_chat_model(
        settings,
        settings.llm_model_fast,
        provider_order=settings.llm_fast_provider_order,
        allow_fallbacks=settings.llm_fast_allow_fallbacks,
    )
    document_meta = load_document_metadata(MANIFEST_PATH)
    claims = load_claims()
    if limit is not None:
        Random(_SHUFFLE_SEED).shuffle(claims)
        claims = claims[:limit]
    chunks = load_chunk_corpus()
    corpus = load_corpus(JSONL_PATH)

    engine = create_engine_from_settings()
    session = create_session_factory(engine=engine)()
    rows: list[_ClaimResult] = []
    errors: list[str] = []
    claim_texts: dict[str, str] = {}
    excerpts: dict[str, dict[str, str]] = {}
    try:
        assert_chunk_table_ready(session)
        adapter = _build_adapter(session, chunks, corpus)
        context = GraphContext(
            fast_model=fast_model,
            reasoning_model=reasoning_model,
            retriever=adapter,
            llm_settings=settings,
        )
        for cohort, claim in claims:
            claim_text = build_claim_text(
                claim, document_meta[claim.document_id], policy_header=policy_header
            )
            claim_texts[claim.claim_id] = claim_text
            started = datetime.now(UTC)
            try:
                with ThreadPoolExecutor(max_workers=1) as pool:
                    state, reached = pool.submit(
                        _run_graph, claim, claim_text, context
                    ).result(timeout=_CLAIM_TIMEOUT_SECONDS)
            except Exception as exc:  # noqa: BLE001 - recorded, run continues
                errors.append(claim.claim_id)
                rows.append(_errored_row(claim, cohort, repr(exc)))
                print(f"{claim.claim_id:<28} ERROR {exc!r}", flush=True)
                continue
            latency = (datetime.now(UTC) - started).total_seconds()
            row = _score_claim(
                claim,
                cohort,
                state,
                reached_checkpoint=reached,
                latency_seconds=latency,
            )
            rows.append(row)
            excerpts[claim.claim_id] = {
                c.clause_id: c.excerpt
                for c in cast(list[Citation], state.get("citations") or [])
            }
            print(
                f"{row.claim_id:<28} cohort={row.cohort:<26} "
                f"expected={row.expected_verdict:<24} "
                f"got={row.predicted_verdict or '-':<24} "
                f"{'OK' if row.correct else 'X'} "
                f"ref_recall={row.reference_recall:.2f} {latency:.0f}s",
                flush=True,
            )

        judge_summary = None
        judge_error: str | None = None
        if judge:
            print("\nJudging faithfulness and context relevance...", flush=True)
            # The judge runs after every claim, so a failure here would otherwise
            # discard a completed run's verdict results -- an hour of graph runs
            # lost to a provider hiccup in the measurement's *last* step. The
            # judge is an added metric, not the run; losing it degrades the
            # report, and the report says so rather than pretending it was
            # never asked for.
            try:
                judge_summary = _judge_run(
                    build_judge_model(settings),
                    rows,
                    claim_texts,
                    excerpts,
                    passes=judge_passes,
                )
            except Exception as exc:  # noqa: BLE001 - recorded, results still written
                judge_error = repr(exc)
                print(f"JUDGE FAILED, verdict results kept: {exc!r}", flush=True)
    finally:
        session.close()
        engine.dispose()

    meta = _build_meta(
        settings=settings,
        claims=claims,
        policy_header=policy_header,
        judge=judge,
        judge_passes=judge_passes,
        limit=limit,
    )
    meta["judge_error"] = judge_error
    return _summarise(rows, errors, judge_summary, meta=meta)


def _errored_row(claim: SyntheticClaim, cohort: str, error: str) -> _ClaimResult:
    return _ClaimResult(
        claim_id=claim.claim_id,
        cohort=cohort,
        document_id=claim.document_id,
        expected_verdict=claim.expected_verdict.value,
        predicted_verdict=None,
        posture=None,
        confidence=None,
        reached_checkpoint=False,
        completed=False,
        context_sufficient=None,
        clarification_rounds=0,
        clarification_exhausted=False,
        compatibility_verdict=None,
        grounding_degraded=False,
        justification_degraded=False,
        justification_names_a_clause=False,
        assertions=(),
        retrieved_clause_ids=(),
        recommendation_citation_ids=(),
        reference_clause_ids=tuple(claim.reference_clause_ids),
        n_consistency_flags=0,
        latency_seconds=0.0,
        error=error,
    )


def _build_meta(
    *,
    settings: Any,
    claims: Sequence[tuple[str, SyntheticClaim]],
    policy_header: bool,
    judge: bool,
    judge_passes: int,
    limit: int | None,
) -> dict[str, Any]:
    counts = Counter(cohort for cohort, _ in claims)
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "platform": platform.platform(),
        "reasoning_model": settings.llm_model_reasoning,
        "fast_model": settings.llm_model_fast,
        "judge_model": JUDGE_MODEL if judge else None,
        "judge_passes": judge_passes if judge else None,
        "policy_header": policy_header,
        "limit": limit,
        "claim_count": len(claims),
        "cohort_counts": dict(sorted(counts.items())),
        "claim_timeout_seconds": _CLAIM_TIMEOUT_SECONDS,
        "resume_decision": _RESUME_DECISION["decision"],
        "dod_note": (
            "DoD [M4-10] item 1 says 30 synthetic claims; the finalized set is "
            "40 (claims.jsonl) + 11 (product_claim_mismatch.jsonl) = 51, all "
            "scored. Same drift noted by [M4-02] in eval/runs/intake_extraction."
        ),
        "method_note": (
            "The whole compiled graph per claim, resumed past the [M4-09] human "
            "checkpoint with an automatic 'approve'. The scored verdict is the "
            "recommendation node's posture, never a reviewed decision."
        ),
    }


def _summarise(
    rows: list[_ClaimResult],
    errors: list[str],
    judge_summary: _JudgeSummary | None,
    *,
    meta: dict[str, Any],
) -> EndToEndEvalResult:
    scored = [r for r in rows if r.error is None]
    by_cohort: dict[str, VerdictMetrics] = {}
    for cohort in sorted({r.cohort for r in scored}):
        subset = [r for r in scored if r.cohort == cohort]
        by_cohort[cohort] = verdict_metrics(
            [(r.expected_verdict, r.predicted_verdict) for r in subset]
        )

    causes = Counter(
        cause for cause in (failure_cause(r) for r in scored) if cause is not None
    )
    with_reference = [r for r in scored if r.reference_clause_ids]
    total_reference = sum(len(set(r.reference_clause_ids)) for r in with_reference)
    settled = [r for r in scored if r.posture in {"compatible", "incompatible"}]
    return EndToEndEvalResult(
        meta=meta,
        overall=verdict_metrics(
            [(r.expected_verdict, r.predicted_verdict) for r in scored]
        ),
        by_cohort=by_cohort,
        completion_rate=(
            sum(1 for r in rows if r.completed and r.reached_checkpoint) / len(rows)
            if rows
            else 0.0
        ),
        failure_causes={cause: causes.get(cause, 0) for cause in _FAILURE_CAUSES},
        reference_recall_micro=(
            sum(r.reference_hits for r in with_reference) / total_reference
            if total_reference
            else 0.0
        ),
        reference_recall_any_rate=(
            sum(1 for r in with_reference if r.reference_hits) / len(with_reference)
            if with_reference
            else 0.0
        ),
        on_target_document_rate=(
            sum(1 for r in scored if r.on_target_document) / len(scored)
            if scored
            else 0.0
        ),
        justification_names_a_clause_rate=(
            sum(1 for r in settled if r.justification_names_a_clause) / len(settled)
            if settled
            else 0.0
        ),
        judge=judge_summary,
        results=rows,
        error_claim_ids=errors,
    )


def _pct(value: float) -> str:
    return f"{value:.1%}"


def snapshot_payload(result: EndToEndEvalResult) -> dict[str, Any]:
    """The committed citation-coverage snapshot the CI validation script reads.

    Carries the assertion text as well as the ids: the check is structural, but
    the artefact is also the published evidence for it, and a reviewer reading
    the file should see what was asserted, not only that something was.
    """
    return {
        "provenance": {
            "generated_by": "scripts/eval_end_to_end.py",
            "command": "make eval-end-to-end",
            "schema_version": SCHEMA_VERSION,
            "generated_at_utc": result.meta["generated_at_utc"],
            "reasoning_model": result.meta["reasoning_model"],
            "fast_model": result.meta["fast_model"],
            "policy_header": result.meta["policy_header"],
            "claim_count": result.meta["claim_count"],
            "note": (
                "Every assertion on a settled verdict must carry >=1 clause id, "
                "and every id here must exist in build/parsed_clauses.jsonl."
            ),
        },
        "claims": [
            {
                "claim_id": r.claim_id,
                "cohort": r.cohort,
                "compatibility_verdict": r.compatibility_verdict,
                "assertions": [
                    {"statement": statement, "clause_ids": list(clause_ids)}
                    for statement, clause_ids in r.assertions
                ],
                "recommendation_citation_ids": list(r.recommendation_citation_ids),
                "retrieved_clause_ids": list(r.retrieved_clause_ids),
            }
            for r in result.results
            if r.error is None
        ],
    }


def _cohort_section(result: EndToEndEvalResult) -> list[str]:
    lines: list[str] = []
    for cohort, metrics in result.by_cohort.items():
        label = (
            "mismatch (product/claim mismatch subset)"
            if cohort == "mismatch"
            else cohort
        )
        lines += [
            "",
            f"#### {label} (n={metrics.n}, accuracy {_pct(metrics.accuracy)})",
            "",
        ]
        lines += confusion_table_lines(metrics)
    return lines


def render_markdown(result: EndToEndEvalResult) -> str:
    """Render the run as Markdown; the numbers are copied into the doc."""
    overall = result.overall
    meta = result.meta
    lines = [
        "# End-to-end verdict accuracy and citation coverage -- measurement ([M4-10])",
        "",
        "Generated by `scripts/eval_end_to_end.py` (`make eval-end-to-end`): the "
        "whole compiled graph over the synthetic claim set "
        f"({meta['claim_count']} claims), reasoning model "
        f"`{meta['reasoning_model']}`, fast model `{meta['fast_model']}`, judge "
        f"`{meta['judge_model'] or 'not run'}`. Regenerable; committed analysis "
        "in `docs/END_TO_END_EVALUATION.md`.",
        "",
        f"- Generated (UTC): {meta['generated_at_utc']}",
        f"- Platform: {meta['platform']}",
        f"- Policy header in the claim text: {meta['policy_header']}",
        f"- Scored: {overall.n} of {meta['claim_count']}",
        f"- Errors: {result.error_claim_ids or 'none'}",
        f"- Note: {meta['dod_note']}",
        "",
        "## Verdict accuracy",
        "",
        f"**Overall accuracy: {_pct(overall.accuracy)}** ({overall.n} scored), "
        f"graph completion {_pct(result.completion_rate)}.",
        "",
        "### Confusion matrix (overall)",
        "",
    ]
    lines += confusion_table_lines(overall)
    lines += ["", "### Per class (overall)", ""]
    lines += per_class_table_lines(overall)
    lines += ["", "### By cohort"]
    lines += _cohort_section(result)
    lines += [
        "",
        "## Failure catalogue",
        "",
        "Every wrong verdict, attributed to one cause (first match, causal order).",
        "",
        "| cause | n |",
        "| --- | ---: |",
    ]
    for cause, count in result.failure_causes.items():
        lines.append(f"| {cause} | {count} |")
    lines += [
        "",
        "## Retrieval reaching the labelled clauses",
        "",
        f"- Reference-clause recall (micro): {_pct(result.reference_recall_micro)}",
        f"- Claims retrieving at least one reference clause: "
        f"{_pct(result.reference_recall_any_rate)}",
        f"- Claims retrieving anything from their own policy document: "
        f"{_pct(result.on_target_document_rate)}",
        "",
        "## Justification",
        "",
        f"- Names a clause id inline on a settled verdict: "
        f"{_pct(result.justification_names_a_clause_rate)}",
        "",
    ]
    if result.judge is None and meta.get("judge_error"):
        lines += [
            "## Judge",
            "",
            f"**The judge did not complete: `{meta['judge_error']}`.** The verdict "
            "numbers above are unaffected -- the judge runs after every claim is "
            "scored -- but faithfulness and context relevance are missing from "
            "this run.",
            "",
        ]
    if result.judge is not None:
        judge = result.judge
        lines += [
            "## Judge (faithfulness and context relevance)",
            "",
            f"Model `{meta['judge_model']}`, {judge.passes} passes per item.",
            "",
            "| metric | value | unanimous | per-pass spread |",
            "| --- | ---: | ---: | ---: |",
            f"| faithfulness (supported) | {_pct(judge.faithfulness_rate)} | "
            f"{_pct(judge.faithfulness_unanimous_rate)} | "
            f"{judge.faithfulness_spread:.3f} |",
            f"| context relevance | {_pct(judge.context_relevance_rate)} | "
            f"{_pct(judge.context_unanimous_rate)} | "
            f"{judge.context_relevance_spread:.3f} |",
            "",
            f"- Assertions judged: {judge.n_items}; clauses judged: "
            f"{judge.n_clauses_judged}",
            f"- Partially supported: {_pct(judge.partially_supported_rate)}; "
            f"unsupported: {_pct(judge.unsupported_rate)}",
            f"- Faithfulness per pass: "
            f"{[round(v, 3) for v in judge.per_pass_faithfulness]}",
            f"- Context relevance per pass: "
            f"{[round(v, 3) for v in judge.per_pass_context_relevance]}",
            "",
        ]
    lines += ["## Findings", "", "_To be written from the first run._", ""]
    return "\n".join(lines) + "\n"


def _prediction_row(r: _ClaimResult) -> dict[str, Any]:
    return {
        "claim_id": r.claim_id,
        "cohort": r.cohort,
        "document_id": r.document_id,
        "expected_verdict": r.expected_verdict,
        "predicted_verdict": r.predicted_verdict,
        "posture": r.posture,
        "correct": r.correct,
        "failure_cause": failure_cause(r),
        "confidence": r.confidence,
        "completed": r.completed,
        "reached_checkpoint": r.reached_checkpoint,
        "context_sufficient": r.context_sufficient,
        "clarification_rounds": r.clarification_rounds,
        "clarification_exhausted": r.clarification_exhausted,
        "compatibility_verdict": r.compatibility_verdict,
        "grounding_degraded": r.grounding_degraded,
        "justification_degraded": r.justification_degraded,
        "justification_names_a_clause": r.justification_names_a_clause,
        "assertions": [
            {"statement": s, "clause_ids": list(ids)} for s, ids in r.assertions
        ],
        "retrieved_clause_ids": list(r.retrieved_clause_ids),
        "recommendation_citation_ids": list(r.recommendation_citation_ids),
        "reference_clause_ids": list(r.reference_clause_ids),
        "reference_recall": r.reference_recall,
        "on_target_document": r.on_target_document,
        "retrieved_document_ids": sorted(r.retrieved_document_ids),
        "n_consistency_flags": r.n_consistency_flags,
        "latency_seconds": r.latency_seconds,
        "error": r.error,
    }


def main() -> None:
    """Run the eval and write ``eval/runs/<stem>.{md,json}`` + predictions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy-header",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="prepend the registered policy's SUSEP process to the claim text",
    )
    parser.add_argument(
        "--judge",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="run the faithfulness / context-relevance judge",
    )
    parser.add_argument(
        "--judge-passes",
        type=int,
        default=JUDGE_PASSES,
        help=f"judge passes per item (default {JUDGE_PASSES})",
    )
    parser.add_argument(
        "--out-stem", default=DEFAULT_STEM, help="basename under eval/runs/"
    )
    parser.add_argument(
        "--write-snapshot",
        action="store_true",
        help=f"also write the committed {SNAPSHOT_PATH}",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="cap the number of claims (smoke runs)"
    )
    args = parser.parse_args()

    result = run_end_to_end_eval(
        policy_header=args.policy_header,
        judge=args.judge,
        limit=args.limit,
        judge_passes=args.judge_passes,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / f"{args.out_stem}.json"
    md_path = OUTPUT_DIR / f"{args.out_stem}.md"
    predictions_path = OUTPUT_DIR / f"{args.out_stem}_predictions.jsonl"
    json_path.write_text(
        json.dumps(result.to_json(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(render_markdown(result), encoding="utf-8")
    with predictions_path.open("w", encoding="utf-8") as handle:
        for row in result.results:
            handle.write(json.dumps(_prediction_row(row), ensure_ascii=False) + "\n")
    written = [str(json_path), str(md_path), str(predictions_path)]
    if args.write_snapshot:
        SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
        SNAPSHOT_PATH.write_text(
            json.dumps(snapshot_payload(result), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        written.append(str(SNAPSHOT_PATH))

    print("")
    print(
        f"accuracy {_pct(result.overall.accuracy)} ({result.overall.n} scored) | "
        f"completion {_pct(result.completion_rate)} | "
        f"causes {result.failure_causes} | "
        f"errors {result.error_claim_ids or 'none'}"
    )
    print("Wrote " + ", ".join(written))


if __name__ == "__main__":
    main()
