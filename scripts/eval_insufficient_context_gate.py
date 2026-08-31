#!/usr/bin/env python3
"""Calibrate and score the [M3-07] insufficient-context gate.

The [M3-07] DoD: "define the gate on retrieval signals ... calibrate the
threshold on the unanswerable subset from [M2-05] ... report precision and
recall of the gate, and the false-negative cases". The M3 exit criteria this
owns (``MILESTONES.md``): **gate precision on the unanswerable subset >= 80%
(reference; if below, document the cause)** and **gate recall on the
unanswerable subset 100%, enforced by an automated test**.

This runs the committed retrieval pipeline -- hybrid RRF + cross-encoder rerank,
filtered to each question's SUSEP process + CNPJ (the ``docs/HYBRID_RETRIEVAL.md``
/ ``docs/RERANKING.md`` base) -- over **all 140** ``golden-set-v1`` questions,
builds an [infrastructure.rag.insufficient_context_gate.GateSignals] per
question, and treats the 23 ``unanswerable`` questions as the positive class and
the 117 scorable ones as the negative class. It sweeps the abstain threshold,
reports the precision/recall curve and the operating point, and writes:

* ``eval/runs/insufficient_context_gate.{md,json}`` -- the full run, gitignored
  and regenerable; the committed curve and the verdict live in
  ``docs/INSUFFICIENT_CONTEXT_GATE.md``;
* ``eval/insufficient_context_gate_signals.json`` -- a **committed** per-question
  signal snapshot the always-on unit test
  (``tests/unit/infrastructure/rag/test_insufficient_context_gate.py``) replays
  to enforce the 100%-recall guarantee without a running stack.

Calibrate on **hybrid + rerank, not + co-retrieval**: exclusion co-retrieval is
a structural post-process that never re-scores, so the rank-1 reranker score --
the gate's signal -- is byte-identical with or without it.

Needs a running Postgres with loaded + embedded chunks and the optional ``embed``
uv group (the reranker + query embedder). Run via
``make eval-insufficient-context-gate``.
"""

from __future__ import annotations

import json
import platform
import statistics
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from domain.clause_classification import ClauseType
from infrastructure.database import (
    assert_chunk_table_ready,
    create_engine_from_settings,
    create_session_factory,
)
from infrastructure.evaluation.golden_set_schema import GoldenQuestion, QuestionType
from infrastructure.parsing.clause_schema import ParsedClauseRecord
from infrastructure.parsing.corpus_artifact import JSONL_PATH
from infrastructure.rag.dense_retriever import DenseRetriever
from infrastructure.rag.embedding_cache import CachingEmbedder
from infrastructure.rag.hybrid_retriever import HybridRetriever
from infrastructure.rag.insufficient_context_config import (
    INSTANCE_VALUE_TOP_SCORE_THRESHOLD,
    TOP_SCORE_ABSTAIN_THRESHOLD,
)
from infrastructure.rag.insufficient_context_config import (
    config_fingerprint as gate_config_fingerprint,
)
from infrastructure.rag.insufficient_context_gate import (
    GateSignals,
    MissingFactCategory,
    classify_missing_information,
    evaluate_gate,
    needs_verified_instance_value,
)
from infrastructure.rag.reranker_cache import CachingReranker
from infrastructure.rag.reranker_config import (
    RERANK_CANDIDATE_DEPTH,
    RERANKER_MODEL_ID,
    RERANKER_MODEL_REVISION,
)
from infrastructure.rag.reranker_config import (
    config_fingerprint as reranker_config_fingerprint,
)
from infrastructure.rag.retrieval_filter import RetrievalFilter
from scripts.eval_retrieval import (
    GOLDEN_SET_DIR,
    MANIFEST_PATH,
    _hybrid_config_fields,
    _load_query_embedder,
    _load_reranker,
    build_clause_text_map,
    build_lexical_retriever,
    load_chunk_corpus,
    load_corpus,
    load_document_metadata,
    load_golden_questions,
)

SCHEMA_VERSION = "v1"
OUTPUT_DIR = Path("eval/runs")
JSON_PATH = OUTPUT_DIR / "insufficient_context_gate.json"
MD_PATH = OUTPUT_DIR / "insufficient_context_gate.md"
SNAPSHOT_PATH = Path("eval/insufficient_context_gate_signals.json")

# The 3 [M2-05] "decoy" unanswerable questions: a real, topically-adjacent
# clause carrying a number that does *not* answer the general policy-level
# question (scripts/unanswerable_question_selection.py MANUAL_DECOY_SPECS).
# These are expected to be the hard cases that push the 100%-recall threshold
# up -- called out here so the report can split them from the clean-absent 20.
DECOY_QUESTION_IDS = ("unanswerable-002", "unanswerable-003", "unanswerable-011")

# The readable grid for the primary-floor (`TOP_SCORE_ABSTAIN_THRESHOLD`) sweep;
# the instance-value floor is held at its pinned value for every row.
THRESHOLD_GRID: tuple[float, ...] = tuple(round(0.30 + 0.025 * i, 3) for i in range(15))


class QuestionSignals(BaseModel):
    """One question's label + the retrieval signals the gate sees."""

    question_id: str
    question_type: str
    question: str
    is_unanswerable: bool
    top_score: float
    reranked_scores: list[float]
    retrieved_clause_ids: list[str]
    retrieved_clause_types: list[str]
    n_returned: int
    k_requested: int

    def to_signals(self) -> GateSignals:
        """Rebuild the frozen [GateSignals] the gate decides on."""
        return GateSignals(
            top_score=self.top_score,
            reranked_scores=tuple(self.reranked_scores),
            retrieved_clause_ids=tuple(self.retrieved_clause_ids),
            retrieved_clause_types=tuple(
                ClauseType(value) for value in self.retrieved_clause_types
            ),
            k_requested=self.k_requested,
            n_returned=self.n_returned,
        )


class InsufficientContextGateConfig(BaseModel):
    """The reproducibility stamp for one ``make eval-insufficient-context-gate`` run."""

    schema_version: str
    run_at_utc: datetime
    golden_set_dir: str
    golden_set_question_count: int
    unanswerable_count: int
    scorable_count: int
    corpus_path: str
    corpus_clause_count: int
    rerank_candidate_depth: int
    reranker_model_id: str
    reranker_model_revision: str
    reranker_config_fingerprint: str
    embedding_config_fingerprint: str
    lexical_config_fingerprint: str
    hybrid_config_fingerprint: str
    gate_config_fingerprint: str
    chosen_top_score_threshold: float
    chosen_instance_value_threshold: float
    scoring_device: str
    platform: str


def collect_signals(
    questions: Sequence[GoldenQuestion],
    document_meta: dict[str, dict[str, str]],
    clause_by_id: dict[str, ParsedClauseRecord],
) -> tuple[list[QuestionSignals], str]:
    """Run hybrid + rerank over every question; return (signals, scoring_device).

    Built directly rather than through ``scripts.eval_retrieval._open_retriever``
    /``evaluate_questions``: that path drops the reranker scores and skips
    ``unanswerable`` questions entirely, and those 23 are the positive class
    here. Mirrors ``scripts/tune_reranking._rerank_at_max_depth``.
    """
    chunks = load_chunk_corpus()
    text_map = build_clause_text_map(chunks)

    real_reranker = _load_reranker()
    scoring_device = str(getattr(real_reranker, "device", "unknown"))
    reranker = CachingReranker(real_reranker)

    engine = create_engine_from_settings()
    session = create_session_factory(engine=engine)()
    rows: list[QuestionSignals] = []
    try:
        assert_chunk_table_ready(session)
        embedder = CachingEmbedder(_load_query_embedder())
        dense = DenseRetriever(session, embedder)
        hybrid = HybridRetriever(build_lexical_retriever(chunks), dense)

        for question in questions:
            metadata_filter = RetrievalFilter.from_manifest_row(
                document_meta[question.document_id]
            )
            candidates = hybrid.retrieve(
                question.question,
                k=RERANK_CANDIDATE_DEPTH,
                metadata_filter=metadata_filter,
            )
            passages = [text_map.get(clause_id, "") for clause_id in candidates]
            scores = reranker.rerank(question.question, passages) if passages else []
            order = sorted(
                range(len(candidates)), key=lambda i: scores[i], reverse=True
            )
            ranked_ids = [candidates[i] for i in order]
            ranked_scores = [float(scores[i]) for i in order]
            rows.append(
                QuestionSignals(
                    question_id=question.question_id,
                    question_type=question.question_type.value,
                    question=question.question,
                    is_unanswerable=(
                        question.question_type is QuestionType.UNANSWERABLE
                    ),
                    top_score=ranked_scores[0] if ranked_scores else 0.0,
                    reranked_scores=ranked_scores,
                    retrieved_clause_ids=ranked_ids,
                    retrieved_clause_types=[
                        clause_by_id[cid].clause_type.value
                        for cid in ranked_ids
                        if cid in clause_by_id
                    ],
                    n_returned=len(ranked_ids),
                    k_requested=RERANK_CANDIDATE_DEPTH,
                )
            )
            print(
                f"{question.question_id:<24} "
                f"{'UNANS' if rows[-1].is_unanswerable else 'scor.':<6} "
                f"top={rows[-1].top_score:.3f} n={rows[-1].n_returned}",
                flush=True,
            )
    finally:
        session.close()
        engine.dispose()
    return rows, scoring_device


def confusion_at(
    rows: Sequence[QuestionSignals],
    *,
    low: float,
    high: float,
) -> dict[str, Any]:
    """Binary abstain/answer confusion over the 140, positive class = unanswerable.

    ``evaluate_gate`` with ``top_score_threshold=low`` and
    ``instance_value_threshold=high`` is the decision; "abstain" ==
    ``not result.sufficient``. ``high=0.0`` disables the instance-value rule.
    """
    tp = fp = fn = tn = 0
    fp_ids: list[str] = []
    fn_ids: list[str] = []
    for row in rows:
        abstained = not evaluate_gate(
            row.question,
            row.to_signals(),
            top_score_threshold=low,
            instance_value_threshold=high,
        ).sufficient
        if row.is_unanswerable and abstained:
            tp += 1
        elif row.is_unanswerable and not abstained:
            fn += 1
            fn_ids.append(row.question_id)
        elif not row.is_unanswerable and abstained:
            fp += 1
            fp_ids.append(f"{row.question_id} ({row.question_type})")
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "low": low,
        "high": high,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "fp_ids": fp_ids,
        "fn_ids": fn_ids,
    }


def pure_floor_sweep(rows: Sequence[QuestionSignals]) -> list[dict[str, Any]]:
    """The primary floor with rule (c) disabled -- can a pure top-score gate work?

    Each grid threshold plus the exact 100%-recall point (just above the highest
    ``unanswerable`` top-score). ``high=0.0`` turns rule (c) off.
    """
    u_max = max(r.top_score for r in rows if r.is_unanswerable)
    thresholds = sorted({*THRESHOLD_GRID, round(u_max + 0.001, 4)})
    return [confusion_at(rows, low=t, high=0.0) for t in thresholds]


def operating_point(rows: Sequence[QuestionSignals]) -> dict[str, Any]:
    """Confusion at the pinned (low, high), plus the per-rule attribution."""
    low = TOP_SCORE_ABSTAIN_THRESHOLD
    high = INSTANCE_VALUE_TOP_SCORE_THRESHOLD
    result = confusion_at(rows, low=low, high=high)

    caught_by_low: list[str] = []
    caught_by_instance_value: list[tuple[str, float]] = []
    missed: list[str] = []
    for row in rows:
        if not row.is_unanswerable:
            continue
        if row.n_returned == 0 or row.top_score < low:
            caught_by_low.append(row.question_id)
        elif row.top_score < high and needs_verified_instance_value(row.question):
            caught_by_instance_value.append((row.question_id, round(row.top_score, 3)))
        else:
            missed.append(row.question_id)

    return {
        **result,
        "caught_by_low_floor": caught_by_low,
        "caught_by_instance_value_rule": caught_by_instance_value,
        "unanswerable_missed": missed,
    }


def fragility(rows: Sequence[QuestionSignals]) -> dict[str, Any]:
    """How wide each threshold's calibration gap is on ``golden-set-v1``."""
    low = TOP_SCORE_ABSTAIN_THRESHOLD
    high = INSTANCE_VALUE_TOP_SCORE_THRESHOLD

    # The primary floor sits between the highest unanswerable NOT flagged as an
    # instance-value question and the lowest answerable top-score.
    low_side_unanswerable = max(
        (
            r.top_score
            for r in rows
            if r.is_unanswerable and not needs_verified_instance_value(r.question)
        ),
        default=0.0,
    )
    lowest_answerable = min(
        (r.top_score for r in rows if not r.is_unanswerable), default=1.0
    )
    # The instance-value floor sits between the highest instance-value
    # unanswerable and the lowest answerable question that also asks for such a
    # value (its answer really is in the corpus).
    high_side_unanswerable = max(
        (
            r.top_score
            for r in rows
            if r.is_unanswerable and needs_verified_instance_value(r.question)
        ),
        default=0.0,
    )
    answerable_instance_value = sorted(
        (round(r.top_score, 3), r.question_id)
        for r in rows
        if not r.is_unanswerable and needs_verified_instance_value(r.question)
    )
    lowest_answerable_instance_value = (
        answerable_instance_value[0][0] if answerable_instance_value else 1.0
    )
    return {
        "low_floor": low,
        "low_floor_gap": [round(low_side_unanswerable, 3), round(lowest_answerable, 3)],
        "instance_value_floor": high,
        "instance_value_floor_gap": [
            round(high_side_unanswerable, 3),
            round(lowest_answerable_instance_value, 3),
        ],
        "answerable_asking_for_an_instance_value": answerable_instance_value,
    }


def _score_stats(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"n": 0, "min": 0.0, "mean": 0.0, "max": 0.0}
    return {
        "n": len(values),
        "min": round(min(values), 3),
        "mean": round(statistics.fmean(values), 3),
        "max": round(max(values), 3),
    }


def decoy_split(rows: Sequence[QuestionSignals]) -> dict[str, dict[str, float]]:
    """Top-score stats for the 3 decoy vs the 20 clean-absent unanswerable Qs."""
    decoy = [r.top_score for r in rows if r.question_id in DECOY_QUESTION_IDS]
    clean = [
        r.top_score
        for r in rows
        if r.is_unanswerable and r.question_id not in DECOY_QUESTION_IDS
    ]
    answerable = [r.top_score for r in rows if not r.is_unanswerable]
    return {
        "decoy": _score_stats(decoy),
        "clean_absent": _score_stats(clean),
        "answerable": _score_stats(answerable),
    }


def category_split(rows: Sequence[QuestionSignals]) -> dict[str, dict[str, float]]:
    """Top-score stats per MissingFactCategory over the 23 unanswerable Qs."""
    buckets: dict[str, list[float]] = {c.value: [] for c in MissingFactCategory}
    for row in rows:
        if not row.is_unanswerable:
            continue
        buckets[classify_missing_information(row.question).value].append(row.top_score)
    return {name: _score_stats(vals) for name, vals in buckets.items() if vals}


def clause_type_diagnostic(
    rows: Sequence[QuestionSignals],
) -> dict[str, int]:
    """How often each clause type appears in the top-k for the 23 unanswerable Qs.

    Reported as a diagnostic only: clause-type coverage is not shippable as a
    runtime gate signal (no query-intent model before M4), see
    ``docs/INSUFFICIENT_CONTEXT_GATE.md``.
    """
    counts: dict[str, int] = {}
    for row in rows:
        if not row.is_unanswerable:
            continue
        for clause_type in row.retrieved_clause_types:
            counts[clause_type] = counts.get(clause_type, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def pooled_top_score(rows: Sequence[QuestionSignals]) -> dict[str, float]:
    """Min / p50 / max of every question's top score."""
    scores = sorted(r.top_score for r in rows)
    return {
        "min": round(scores[0], 3),
        "p50": round(statistics.median(scores), 3),
        "max": round(scores[-1], 3),
    }


def build_report(
    rows: Sequence[QuestionSignals],
    config: InsufficientContextGateConfig,
) -> dict[str, Any]:
    """Assemble the one dict both the JSON and Markdown outputs render from."""
    return {
        "config": config.model_dump(mode="json"),
        "pure_floor_sweep": pure_floor_sweep(rows),
        "operating_point": operating_point(rows),
        "fragility": fragility(rows),
        "decoy_split": decoy_split(rows),
        "category_split": category_split(rows),
        "clause_type_diagnostic": clause_type_diagnostic(rows),
        "pooled_top_score": pooled_top_score(rows),
    }


def _pct(value: float) -> str:
    return f"{value:.1%}"


def render_markdown_report(report: dict[str, Any]) -> str:
    """Render the gate calibration as Markdown (numbers copied into the doc)."""
    config = report["config"]
    op = report["operating_point"]
    frag = report["fragility"]
    lines = [
        "# Insufficient-context gate calibration",
        "",
        "Generated by `scripts/eval_insufficient_context_gate.py` "
        "(`make eval-insufficient-context-gate`) against the 140 `golden-set-v1` "
        "questions, hybrid RRF + cross-encoder rerank, `--filter default` "
        "(SUSEP process + CNPJ). Positive class = the 23 `unanswerable` "
        "questions; negative class = the 117 scorable ones. Regenerable; the "
        "committed curve and the verdict live in "
        "`docs/INSUFFICIENT_CONTEXT_GATE.md`.",
        "",
        "## Run configuration",
        "",
        f"- Golden set: `{config['golden_set_dir']}` "
        f"({config['golden_set_question_count']} questions; "
        f"{config['unanswerable_count']} unanswerable / "
        f"{config['scorable_count']} scorable)",
        f"- Corpus: `{config['corpus_path']}` "
        f"({config['corpus_clause_count']} clauses)",
        f"- Reranker: `{config['reranker_model_id']}` @ "
        f"`{config['reranker_model_revision']}` "
        f"(fingerprint `{config['reranker_config_fingerprint']}`); "
        f"candidate depth {config['rerank_candidate_depth']}",
        f"- Hybrid RRF fingerprint: `{config['hybrid_config_fingerprint']}`; "
        f"embedding `{config['embedding_config_fingerprint']}`; "
        f"lexical `{config['lexical_config_fingerprint']}`",
        f"- Gate config fingerprint: `{config['gate_config_fingerprint']}`; "
        f"pinned `TOP_SCORE_ABSTAIN_THRESHOLD` = "
        f"**{config['chosen_top_score_threshold']}**, "
        f"`INSTANCE_VALUE_TOP_SCORE_THRESHOLD` = "
        f"**{config['chosen_instance_value_threshold']}**",
        f"- Scoring device: `{config['scoring_device']}`; "
        f"platform: {config['platform']}",
        f"- Run at (UTC): {config['run_at_utc']}",
        "",
        "## The rule",
        "",
        "`evaluate_gate` abstains when: (a) nothing was retrieved, **or** "
        "(b) the rank-1 reranked score is below `TOP_SCORE_ABSTAIN_THRESHOLD` "
        f"(**{config['chosen_top_score_threshold']}**), **or** (c) the question "
        "asks for a specific policy-instance value of a `docs/SCOPE.md`-absent "
        "fact (`needs_verified_instance_value`) and the rank-1 score is below "
        f"`INSTANCE_VALUE_TOP_SCORE_THRESHOLD` "
        f"(**{config['chosen_instance_value_threshold']}**).",
        "",
        "## Can a pure top-score gate work? (rule c disabled)",
        "",
        "| `TOP_SCORE_ABSTAIN_THRESHOLD` | recall | precision | TP | FP | FN |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for point in report["pure_floor_sweep"]:
        lines.append(
            f"| {point['low']:.3f} | {_pct(point['recall'])} "
            f"| {_pct(point['precision'])} | {point['tp']} | {point['fp']} "
            f"| {point['fn']} |"
        )
    lines += [
        "",
        "## Operating point — the pinned (low, high)",
        "",
        f"- Recall **{_pct(op['recall'])}** ({op['tp']}/23) — the [M3-07] DoD "
        "guarantee, also locked by a unit test over the committed snapshot",
        f"- Precision **{_pct(op['precision'])}** "
        f"({op['tp']}/{op['tp'] + op['fp']}); {op['fp']} false positives "
        "(bar: >= 80% ⇔ FP <= 5)",
        f"- Unanswerable caught by the primary floor: {op['caught_by_low_floor']}",
        "- Unanswerable caught by the instance-value rule: "
        f"{op['caught_by_instance_value_rule']}",
        f"- Unanswerable missed (must be empty): {op['unanswerable_missed']}",
        "",
        "False positives (answerable questions the gate abstains on):",
        "",
    ]
    lines += [f"- {qid}" for qid in op["fp_ids"]] or ["- (none)"]
    lines += [
        "",
        "False negatives at the pinned config (answering when it should abstain "
        "— the expensive error): " + (str(op["fn_ids"]) if op["fn_ids"] else "(none)"),
        "",
        "## Calibration fragility (this is a fit on 23 points, not a held-out result)",
        "",
        f"- Primary floor {frag['low_floor']}: sits between the highest "
        f"floor-only unanswerable top-score {frag['low_floor_gap'][0]} and the "
        f"lowest answerable top-score {frag['low_floor_gap'][1]}",
        f"- Instance-value floor {frag['instance_value_floor']}: sits between the "
        f"highest instance-value unanswerable top-score "
        f"{frag['instance_value_floor_gap'][0]} and the lowest answerable "
        f"instance-value question {frag['instance_value_floor_gap'][1]}",
        "- Answerable questions that also ask for an instance value "
        f"(handled by their score): {frag['answerable_asking_for_an_instance_value']}",
        f"- Pooled top-score min / p50 / max: "
        f"{report['pooled_top_score']['min']} / "
        f"{report['pooled_top_score']['p50']} / "
        f"{report['pooled_top_score']['max']}",
        "",
        "## Decoy vs clean-absent top-score split",
        "",
        "| group | n | min | mean | max |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, stats in report["decoy_split"].items():
        lines.append(
            f"| {name} | {stats['n']} | {stats['min']} | {stats['mean']} "
            f"| {stats['max']} |"
        )
    lines += [
        "",
        "## Top-score by missing-fact category (unanswerable subset)",
        "",
        "| category | n | min | mean | max |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, stats in report["category_split"].items():
        lines.append(
            f"| {name} | {stats['n']} | {stats['min']} | {stats['mean']} "
            f"| {stats['max']} |"
        )
    lines += [
        "",
        "## Clause-type diagnostic (unanswerable top-k; not a gate signal)",
        "",
        f"{report['clause_type_diagnostic']}",
        "",
    ]
    return "\n".join(lines)


def write_snapshot(
    rows: Sequence[QuestionSignals], config: InsufficientContextGateConfig
) -> None:
    """Write the committed per-question signal snapshot the unit test replays.

    Deliberately carries no timestamp: the reranker scores are
    device-independent and cached, so re-running this over an unchanged corpus /
    golden set / reranker config produces a byte-identical file. The timestamped
    full config lives in the gitignored ``eval/runs/`` run.
    """
    payload = {
        "provenance": {
            "generated_by": "scripts/eval_insufficient_context_gate.py",
            "rerank_candidate_depth": config.rerank_candidate_depth,
            "reranker_config_fingerprint": config.reranker_config_fingerprint,
            "hybrid_config_fingerprint": config.hybrid_config_fingerprint,
            "gate_config_fingerprint": config.gate_config_fingerprint,
            "chosen_top_score_threshold": config.chosen_top_score_threshold,
            "chosen_instance_value_threshold": (config.chosen_instance_value_threshold),
        },
        "questions": [row.model_dump() for row in rows],
    }
    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main() -> None:
    """Run the calibration and write the run + the committed snapshot."""
    document_meta = load_document_metadata(MANIFEST_PATH)
    corpus = load_corpus(JSONL_PATH)
    clause_by_id = {record.clause_id: record for record in corpus}
    questions = load_golden_questions(GOLDEN_SET_DIR)
    chunks = load_chunk_corpus()
    hybrid_fields = _hybrid_config_fields(chunks, "rrf")

    rows, scoring_device = collect_signals(questions, document_meta, clause_by_id)
    unanswerable = [r for r in rows if r.is_unanswerable]
    scorable = [r for r in rows if not r.is_unanswerable]

    config = InsufficientContextGateConfig(
        schema_version=SCHEMA_VERSION,
        run_at_utc=datetime.now(UTC),
        golden_set_dir=str(GOLDEN_SET_DIR),
        golden_set_question_count=len(questions),
        unanswerable_count=len(unanswerable),
        scorable_count=len(scorable),
        corpus_path=str(JSONL_PATH),
        corpus_clause_count=len(corpus),
        rerank_candidate_depth=RERANK_CANDIDATE_DEPTH,
        reranker_model_id=RERANKER_MODEL_ID,
        reranker_model_revision=RERANKER_MODEL_REVISION,
        reranker_config_fingerprint=reranker_config_fingerprint(),
        embedding_config_fingerprint=hybrid_fields["embedding_config_fingerprint"],
        lexical_config_fingerprint=hybrid_fields["lexical_config_fingerprint"],
        hybrid_config_fingerprint=hybrid_fields["hybrid_config_fingerprint"],
        gate_config_fingerprint=gate_config_fingerprint(),
        chosen_top_score_threshold=TOP_SCORE_ABSTAIN_THRESHOLD,
        chosen_instance_value_threshold=INSTANCE_VALUE_TOP_SCORE_THRESHOLD,
        scoring_device=scoring_device,
        platform=platform.platform(),
    )

    report = build_report(rows, config)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    MD_PATH.write_text(render_markdown_report(report), encoding="utf-8")
    write_snapshot(rows, config)

    op = report["operating_point"]
    print("")
    print(
        f"pinned (low, high) = "
        f"({TOP_SCORE_ABSTAIN_THRESHOLD}, {INSTANCE_VALUE_TOP_SCORE_THRESHOLD})"
    )
    print(
        f"recall {_pct(op['recall'])} ({op['tp']}/23); "
        f"precision {_pct(op['precision'])} ({op['fp']} FP; bar is FP <= 5)"
    )
    print(f"unanswerable missed (must be []): {op['unanswerable_missed']}")
    print(f"Wrote {JSON_PATH}, {MD_PATH} and {SNAPSHOT_PATH}")


if __name__ == "__main__":
    main()
