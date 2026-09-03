#!/usr/bin/env python3
"""Measure the [M4-07] wall-clock gain of the fixed parallel assessment branches.

[M4-07] wires the compatibility ([M4-05]) and consistency ([M4-06]) nodes as
fixed parallel branches off the retrieval node -- one superstep, both nodes --
converging on ``END`` (the recommendation node once [M4-08] lands). The DoD:
"measure the wall-clock gain against a sequential run and record it".

This script isolates the assessment stage. Per synthetic claim it builds a
pre-populated ``ClaimState`` (entities from the claim narrative + the document's
own product line / SUSEP process, citations from that document's coverage and
exclusion chunks -- no Postgres, no ANN retriever: the retrieval step is
identical in both modes and irrelevant to the comparison), then runs the exact
[M4-07] fan-out/fan-in -- a minimal compiled ``StateGraph`` with
``START -> {compatibility, consistency} -> END`` -- **once**, with each node
wrapped to record its own wall time.

* **parallel** = the measured wall of that single ``.invoke`` (the two nodes run
  in one superstep, on the sync runner's thread pool).
* **sequential** = ``t_compatibility + t_consistency``, the two node times the
  same run recorded -- what the identical work costs run back to back.

Each LLM call is issued exactly once, so there is no run-to-run variance (the
compatibility node's grounding-retry count differs between calls) and no
provider prompt-cache artifact (a second identical prompt returns faster) to
distort the comparison. The gain is ``1 - parallel / sequential`` and is bounded
by ``min(t_compat, t_consist)``, which the per-node means make visible.

Parity: the single run must populate both ``compatibility`` and ``consistency``
and its ``audit_trail`` must carry one compatibility event and two consistency
events -- verdict text is LLM-nondeterministic, so no exact-output assert.

Needs ``LLM_*`` in ``.env`` (both the fast and reasoning models) and
``build/chunks.jsonl`` (``make build-chunks``). No retrieval stack. Run via
``make eval-parallel-assessment`` (``LLM_PROVIDER=openai make
eval-parallel-assessment`` locally -- the ``.env`` leaves ``LLM_PROVIDER``
blank). Writes ``eval/runs/parallel_assessment.{md,json}`` and a per-claim
``eval/runs/parallel_assessment_timings.jsonl``; the committed analysis lives in
``docs/PARALLEL_ASSESSMENT.md``.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
from collections import defaultdict
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from random import Random
from time import perf_counter
from typing import Any, cast

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from domain.clause_classification import ClauseType
from infrastructure.config.llm_client_factory import build_chat_model
from infrastructure.config.settings import get_llm_settings
from infrastructure.evaluation.synthetic_claims_schema import SyntheticClaim
from infrastructure.graph.context import GraphContext
from infrastructure.graph.nodes.compatibility import compatibility
from infrastructure.graph.nodes.consistency import consistency
from infrastructure.graph.state import (
    Citation,
    ClaimState,
    ExtractedEntities,
)
from infrastructure.rag.chunk_artifact import CHUNKS_JSONL_PATH, read_chunks_jsonl
from scripts.eval_consistency import load_claims

SCHEMA_VERSION = "v1"
OUTPUT_DIR = Path("eval/runs")
JSON_PATH = OUTPUT_DIR / "parallel_assessment.json"
MD_PATH = OUTPUT_DIR / "parallel_assessment.md"
TIMINGS_PATH = OUTPUT_DIR / "parallel_assessment_timings.jsonl"

# How many clauses to hand the compatibility node -- enough real context for a
# genuine reasoning call, without making the prompt unrepresentatively long.
_MAX_CITATIONS = 10

# Ceiling on one claim's assessment run. A hung provider connection then costs
# that claim an error, not the whole run. Comfortably above the observed
# per-claim wall (compatibility ~45-70s incl. any grounding retry; consistency
# ~20s runs under it), tight enough that a hang is abandoned quickly.
_CLAIM_TIMEOUT_SECONDS = 180.0

# Seed for the claim shuffle, so ``--limit`` samples across cohorts (the claim
# files are grouped by verdict) and a partial run is still representative.
_SHUFFLE_SEED = 0


@dataclass(frozen=True)
class _ClaimTiming:
    claim_id: str
    cohort: str
    n_citations: int
    sequential_seconds: float
    parallel_seconds: float
    compatibility_seconds: float
    consistency_seconds: float
    parity_ok: bool
    error: str | None = None

    @property
    def saving_seconds(self) -> float:
        return self.sequential_seconds - self.parallel_seconds

    @property
    def gain_fraction(self) -> float:
        if self.sequential_seconds <= 0:
            return 0.0
        return self.saving_seconds / self.sequential_seconds


@dataclass(frozen=True)
class _Aggregate:
    n_claims: int
    n_errors: int
    n_parity_failures: int
    sequential_total_seconds: float
    parallel_total_seconds: float
    saving_total_seconds: float
    gain_fraction: float
    mean_sequential_seconds: float
    mean_parallel_seconds: float
    mean_compatibility_seconds: float
    mean_consistency_seconds: float


@dataclass(frozen=True)
class ParallelAssessmentEvalResult:
    """Everything ``make eval-parallel-assessment`` produces, for the report + test."""

    meta: dict[str, Any]
    aggregate: _Aggregate
    timings: list[_ClaimTiming]
    error_claim_ids: list[str] = field(default_factory=list)
    parity_failure_claim_ids: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        """The JSON-serialisable view written to ``parallel_assessment.json``."""
        return {
            "schema_version": SCHEMA_VERSION,
            "meta": self.meta,
            "aggregate": {
                "n_claims": self.aggregate.n_claims,
                "n_errors": self.aggregate.n_errors,
                "n_parity_failures": self.aggregate.n_parity_failures,
                "sequential_total_seconds": self.aggregate.sequential_total_seconds,
                "parallel_total_seconds": self.aggregate.parallel_total_seconds,
                "saving_total_seconds": self.aggregate.saving_total_seconds,
                "gain_fraction": self.aggregate.gain_fraction,
                "mean_sequential_seconds": self.aggregate.mean_sequential_seconds,
                "mean_parallel_seconds": self.aggregate.mean_parallel_seconds,
                "mean_compatibility_seconds": self.aggregate.mean_compatibility_seconds,
                "mean_consistency_seconds": self.aggregate.mean_consistency_seconds,
            },
            "error_claim_ids": self.error_claim_ids,
            "parity_failure_claim_ids": self.parity_failure_claim_ids,
        }


class _UnusedRetriever:
    """``GraphContext`` requires a retriever; neither assessment node calls it."""

    def retrieve(
        self, question: str, *, k: int, metadata_filter: object | None = None
    ) -> list[Any]:
        raise AssertionError("the assessment nodes do not retrieve")


def _chunks_by_document() -> dict[str, list[Any]]:
    """Group the chunk corpus by ``document_id`` (fails loudly if not built)."""
    if not CHUNKS_JSONL_PATH.exists():
        raise FileNotFoundError(
            f"{CHUNKS_JSONL_PATH} does not exist. Run `make build-chunks` "
            "(after `make parse` / `make fetch-corpus-artifacts`) first."
        )
    grouped: dict[str, list[Any]] = defaultdict(list)
    for record in read_chunks_jsonl(CHUNKS_JSONL_PATH):
        grouped[record.document_id].append(record)
    return grouped


def _pick_citations(claim: SyntheticClaim, doc_chunks: Sequence[Any]) -> list[Citation]:
    """Realistic context: reference clauses, then coverage, then exclusions."""
    referenced = set(claim.reference_clause_ids)
    ordered: list[Any] = []
    seen: set[str] = set()

    def _add(record: Any) -> None:
        if record.clause_id not in seen and record.display_text.strip():
            ordered.append(record)
            seen.add(record.clause_id)

    for record in doc_chunks:
        if record.clause_id in referenced:
            _add(record)
    for wanted in (ClauseType.COVERAGE, ClauseType.EXCLUSION):
        for record in doc_chunks:
            if record.clause_type is wanted:
                _add(record)
    for record in doc_chunks:
        _add(record)

    ordered = ordered[:_MAX_CITATIONS]
    total = len(ordered)
    return [
        Citation(
            clause_id=record.clause_id,
            document_id=record.document_id,
            susep_process=record.susep_process,
            clause_type=record.clause_type,
            relevance_score=round(1.0 - index / max(total, 1) * 0.5, 4),
            excerpt=record.display_text[:600],
        )
        for index, record in enumerate(ordered)
    ]


def _prepare_state(
    claim: SyntheticClaim, doc_chunks: Sequence[Any]
) -> dict[str, object] | None:
    """Build one claim's pre-assessment ``ClaimState``, or ``None`` if unusable."""
    if not doc_chunks:
        return None
    citations = _pick_citations(claim, doc_chunks)
    if not citations:
        return None
    head = doc_chunks[0]
    entities = ExtractedEntities(
        description=claim.narrative,
        susep_process=head.susep_process,
        product_line=head.product_line,
    )
    return {
        "claim_id": claim.claim_id,
        "raw_claim_text": claim.narrative,
        "entities": entities,
        "citations": citations,
        "context_sufficient": True,
    }


def _timed_node(name: str, fn: Any, durations: dict[str, float]) -> Any:
    """Wrap a node so it records its own wall time into ``durations[name]``."""

    def wrapper(state: ClaimState, runtime: Runtime[GraphContext]) -> dict[str, object]:
        start = perf_counter()
        try:
            return cast("dict[str, object]", fn(state, runtime))
        finally:
            durations[name] = perf_counter() - start

    wrapper.__name__ = name
    return wrapper


def _build_parallel_graph(durations: dict[str, float]) -> Any:
    """The exact [M4-07] fan-out/fan-in, each node wrapped to time itself.

    A fresh ``durations`` dict per claim, so a timed-out claim whose ``.invoke``
    thread is still running cannot scribble into the next claim's timing.
    """
    builder: Any = StateGraph(ClaimState, context_schema=GraphContext)
    builder.add_node(
        "compatibility", _timed_node("compatibility", compatibility, durations)
    )
    builder.add_node("consistency", _timed_node("consistency", consistency, durations))
    builder.add_edge(START, "compatibility")
    builder.add_edge(START, "consistency")
    builder.add_edge("compatibility", END)
    builder.add_edge("consistency", END)
    return builder.compile()


def _audit_nodes(update: dict[str, object]) -> list[str]:
    return [event.node for event in cast(list[Any], update.get("audit_trail") or [])]


def _time_claim(
    cohort: str,
    claim: SyntheticClaim,
    state: dict[str, object],
    context: GraphContext,
) -> _ClaimTiming:
    durations: dict[str, float] = {}
    parallel_graph = _build_parallel_graph(durations)

    par_start = perf_counter()
    par_out = parallel_graph.invoke(dict(state), context=context)
    parallel_seconds = perf_counter() - par_start

    compatibility_seconds = durations.get("compatibility", 0.0)
    consistency_seconds = durations.get("consistency", 0.0)
    # What the identical work costs run back to back -- the two node times the
    # one run recorded, so no retry-count / prompt-cache drift between calls.
    sequential_seconds = compatibility_seconds + consistency_seconds

    parallel_nodes = sorted(_audit_nodes(par_out))
    parity_ok = (
        par_out.get("compatibility") is not None
        and par_out.get("consistency") is not None
        and parallel_nodes == ["compatibility", "consistency", "consistency"]
    )

    return _ClaimTiming(
        claim_id=claim.claim_id,
        cohort=cohort,
        n_citations=len(cast(list[Any], state["citations"])),
        sequential_seconds=round(sequential_seconds, 4),
        parallel_seconds=round(parallel_seconds, 4),
        compatibility_seconds=round(compatibility_seconds, 4),
        consistency_seconds=round(consistency_seconds, 4),
        parity_ok=parity_ok,
    )


def _aggregate(timings: Sequence[_ClaimTiming]) -> _Aggregate:
    ok = [t for t in timings if t.error is None]
    seq_total = sum(t.sequential_seconds for t in ok)
    par_total = sum(t.parallel_seconds for t in ok)
    saving = seq_total - par_total

    def _mean(values: list[float]) -> float:
        return round(statistics.mean(values), 3) if values else 0.0

    return _Aggregate(
        n_claims=len(timings),
        n_errors=sum(1 for t in timings if t.error is not None),
        n_parity_failures=sum(1 for t in ok if not t.parity_ok),
        sequential_total_seconds=round(seq_total, 2),
        parallel_total_seconds=round(par_total, 2),
        saving_total_seconds=round(saving, 2),
        gain_fraction=round(saving / seq_total, 4) if seq_total else 0.0,
        mean_sequential_seconds=_mean([t.sequential_seconds for t in ok]),
        mean_parallel_seconds=_mean([t.parallel_seconds for t in ok]),
        mean_compatibility_seconds=_mean([t.compatibility_seconds for t in ok]),
        mean_consistency_seconds=_mean([t.consistency_seconds for t in ok]),
    )


def run_parallel_assessment_eval(
    limit: int | None = None,
) -> ParallelAssessmentEvalResult:
    """Time the [M4-07] parallel branches vs. the sum of their per-node latencies."""
    settings = get_llm_settings()
    fast_model = build_chat_model(
        settings,
        settings.llm_model_fast,
        provider_order=settings.llm_fast_provider_order,
        allow_fallbacks=settings.llm_fast_allow_fallbacks,
    )
    reasoning_model = build_chat_model(
        settings,
        settings.llm_model_reasoning,
        provider_order=settings.llm_reasoning_provider_order,
        allow_fallbacks=settings.llm_reasoning_allow_fallbacks,
    )
    context = GraphContext(
        fast_model=fast_model,
        reasoning_model=reasoning_model,
        retriever=cast(Any, _UnusedRetriever()),
        llm_settings=settings,
    )

    chunks_by_doc = _chunks_by_document()
    claims = load_claims()
    Random(_SHUFFLE_SEED).shuffle(claims)
    if limit is not None:
        claims = claims[:limit]

    timings: list[_ClaimTiming] = []
    for cohort, claim in claims:
        state = _prepare_state(claim, chunks_by_doc.get(claim.document_id, []))
        if state is None:
            timings.append(
                _ClaimTiming(
                    claim_id=claim.claim_id,
                    cohort=cohort,
                    n_citations=0,
                    sequential_seconds=0.0,
                    parallel_seconds=0.0,
                    compatibility_seconds=0.0,
                    consistency_seconds=0.0,
                    parity_ok=False,
                    error="no usable chunks for document",
                )
            )
            continue
        # A fresh single-worker executor per claim: on a normal finish the thread
        # is joined cleanly; on a provider hang the claim times out, the thread is
        # abandoned (isolated -- its graph and durations dict are its own), and
        # the next claim starts on a new one rather than queueing behind it.
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="claim")
        try:
            timing = executor.submit(_time_claim, cohort, claim, state, context).result(
                timeout=_CLAIM_TIMEOUT_SECONDS
            )
        except FutureTimeoutError:
            timing = _ClaimTiming(
                claim_id=claim.claim_id,
                cohort=cohort,
                n_citations=len(cast(list[Any], state["citations"])),
                sequential_seconds=0.0,
                parallel_seconds=0.0,
                compatibility_seconds=0.0,
                consistency_seconds=0.0,
                parity_ok=False,
                error=f"claim exceeded {_CLAIM_TIMEOUT_SECONDS:.0f}s",
            )
        except Exception as exc:  # noqa: BLE001 - recorded, the run continues
            timing = _ClaimTiming(
                claim_id=claim.claim_id,
                cohort=cohort,
                n_citations=len(cast(list[Any], state["citations"])),
                sequential_seconds=0.0,
                parallel_seconds=0.0,
                compatibility_seconds=0.0,
                consistency_seconds=0.0,
                parity_ok=False,
                error=repr(exc),
            )
        finally:
            executor.shutdown(wait=False)
        timings.append(timing)
        print(
            f"{timing.claim_id:<28} seq={timing.sequential_seconds:>7.2f}s "
            f"par={timing.parallel_seconds:>7.2f}s "
            f"save={timing.saving_seconds:>6.2f}s "
            f"({timing.gain_fraction:>5.1%})"
            + ("" if timing.parity_ok else "  PARITY!")
            + (f"  ERROR {timing.error}" if timing.error else ""),
            flush=True,
        )

    aggregate = _aggregate(timings)
    meta = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "fast_model": settings.llm_model_fast,
        "reasoning_model": settings.llm_model_reasoning,
        "fast_provider_order": settings.llm_fast_provider_order,
        "reasoning_provider_order": settings.llm_reasoning_provider_order,
        "n_claims": len(claims),
        "max_citations": _MAX_CITATIONS,
        "platform": platform.platform(),
        "method_note": (
            "Assessment stage only. citations built from the claim document's "
            "coverage/exclusion chunks (no ANN retriever). One run of the [M4-07] "
            "fan-out/fan-in per claim, each node wrapped to time itself: parallel "
            "= the .invoke wall, sequential = t_compatibility + t_consistency "
            "from that same run. Each LLM call is issued once -- no grounding-"
            "retry drift or prompt-cache artifact between calls. Claims are "
            f"shuffled (seed {_SHUFFLE_SEED}) before --limit so a partial run "
            f"spans cohorts; a claim is abandoned after {_CLAIM_TIMEOUT_SECONDS:.0f}s "
            "(hung provider call) and recorded as an error."
        ),
    }
    return ParallelAssessmentEvalResult(
        meta=meta,
        aggregate=aggregate,
        timings=timings,
        error_claim_ids=[t.claim_id for t in timings if t.error is not None],
        parity_failure_claim_ids=[
            t.claim_id for t in timings if t.error is None and not t.parity_ok
        ],
    )


def _pct(value: float) -> str:
    return f"{value:.1%}"


def render_markdown(result: ParallelAssessmentEvalResult) -> str:
    """Render the run as Markdown; the numbers are copied into the doc."""
    agg = result.aggregate
    meta = result.meta
    lines = [
        "# Parallel assessment branches -- wall-clock measurement ([M4-07])",
        "",
        "Generated by `scripts/eval_parallel_assessment.py` "
        "(`make eval-parallel-assessment`): one run of [M4-07]'s "
        "compatibility ([M4-05]) + consistency ([M4-06]) fan-out/fan-in per "
        "synthetic claim, each node timed. `parallel` = the `.invoke` wall; "
        "`sequential` = the sum of the two node times that same run recorded. "
        "Regenerable; committed analysis in `docs/PARALLEL_ASSESSMENT.md`.",
        "",
        f"- Generated (UTC): {meta['generated_at_utc']}",
        f"- Reasoning model (compatibility): `{meta['reasoning_model']}` "
        f"(provider order `{meta['reasoning_provider_order']}`)",
        f"- Fast model (consistency): `{meta['fast_model']}` "
        f"(provider order `{meta['fast_provider_order']}`)",
        f"- Platform: {meta['platform']}",
        f"- Claims: {agg.n_claims} "
        f"({agg.n_errors} errors {result.error_claim_ids or ''}, "
        f"{agg.n_parity_failures} parity failures "
        f"{result.parity_failure_claim_ids or ''})",
        f"- Method: {meta['method_note']}",
        "",
        "## Wall-clock gain",
        "",
        "| | sequential | parallel | saving |",
        "| --- | ---: | ---: | ---: |",
        f"| total | {agg.sequential_total_seconds:.2f}s | "
        f"{agg.parallel_total_seconds:.2f}s | "
        f"{agg.saving_total_seconds:.2f}s |",
        f"| mean / claim | {agg.mean_sequential_seconds:.3f}s | "
        f"{agg.mean_parallel_seconds:.3f}s | "
        f"{agg.mean_sequential_seconds - agg.mean_parallel_seconds:.3f}s |",
        "",
        f"**Wall-clock gain: {_pct(agg.gain_fraction)}** "
        f"({agg.saving_total_seconds:.2f}s of {agg.sequential_total_seconds:.2f}s).",
        "",
        "### Why",
        "",
        f"- mean compatibility (reasoning model): "
        f"**{agg.mean_compatibility_seconds:.3f}s**",
        f"- mean consistency (fast model): **{agg.mean_consistency_seconds:.3f}s**",
        "",
        "The two run in one superstep, so the parallel wall is about "
        "`max(t_compat, t_consist)` plus overhead and the saving is bounded by "
        "`min(t_compat, t_consist)` -- roughly the consistency (fast-model) call, "
        "which is why the consistency node deliberately uses the fast model "
        "(see `docs/CONSISTENCY_NODE.md`).",
        "",
        "## Findings",
        "",
        "_To be written from the first run._",
        "",
    ]
    return "\n".join(lines) + "\n"


def _timing_row(t: _ClaimTiming) -> dict[str, Any]:
    return {
        "claim_id": t.claim_id,
        "cohort": t.cohort,
        "n_citations": t.n_citations,
        "sequential_seconds": t.sequential_seconds,
        "parallel_seconds": t.parallel_seconds,
        "compatibility_seconds": t.compatibility_seconds,
        "consistency_seconds": t.consistency_seconds,
        "saving_seconds": round(t.saving_seconds, 4),
        "gain_fraction": round(t.gain_fraction, 4),
        "parity_ok": t.parity_ok,
        "error": t.error,
    }


def main() -> None:
    """Run the measurement and write ``eval/runs/parallel_assessment.{md,json}``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit", type=int, default=None, help="cap the number of claims (smoke runs)"
    )
    args = parser.parse_args()

    result = run_parallel_assessment_eval(limit=args.limit)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(
        json.dumps(result.to_json(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    MD_PATH.write_text(render_markdown(result), encoding="utf-8")
    with TIMINGS_PATH.open("w", encoding="utf-8") as handle:
        for timing in result.timings:
            handle.write(json.dumps(_timing_row(timing), ensure_ascii=False) + "\n")

    agg = result.aggregate
    print("")
    print(
        f"wall-clock gain {_pct(agg.gain_fraction)} "
        f"(seq {agg.sequential_total_seconds:.1f}s -> par "
        f"{agg.parallel_total_seconds:.1f}s) | "
        f"errors {result.error_claim_ids or 'none'} | "
        f"parity failures {result.parity_failure_claim_ids or 'none'}"
    )
    print(f"Wrote {JSON_PATH}, {MD_PATH} and {TIMINGS_PATH}")


if __name__ == "__main__":
    main()
