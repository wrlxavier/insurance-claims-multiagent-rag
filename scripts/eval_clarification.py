#!/usr/bin/env python3
"""Measure the clarification loop over the incomplete synthetic claims [M4-03].

DoD item 5 -- "test with the ≥10 incomplete claims from [M2-04]; verify the loop
terminates in every case" -- is a structural property and is covered by
``tests/unit/infrastructure/graph/test_claim_graph.py`` with a fake LLM. This
script is the on-demand live check the other DoD item needs: item 2, "generate a
specific question per missing field, not a generic request for more detail",
which cannot be asserted without seeing real model output.

Two passes over the 13 ``insufficient_information`` claims in
``data/synthetic_claims/claims.jsonl``:

1. Feed each claim's labelled ``missing_fact_type`` straight in as
   ``missing_information`` and call the clarification node once with the real
   fast model -- record the question it generates and flag generic phrasing.
2. Run the full compiled graph (real fast model for intake and clarification)
   -- confirm every claim terminates and report how many actually entered the
   loop and the ``clarification_rounds`` distribution.

Writes ``eval/runs/clarification_loop.{md,json}`` and a per-claim
``eval/runs/clarification_questions.jsonl``.
"""

from __future__ import annotations

import dataclasses
import json
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.runtime import Runtime

from infrastructure.config.llm_client_factory import build_chat_model
from infrastructure.config.settings import get_llm_settings
from infrastructure.evaluation.synthetic_claims_schema import SyntheticClaim
from infrastructure.graph.build import MAX_CLARIFICATION_ROUNDS, build_claim_graph
from infrastructure.graph.context import GraphContext
from infrastructure.graph.nodes.clarification import clarification
from infrastructure.graph.state import ClaimState

CLAIMS_PATH = Path("data/synthetic_claims/claims.jsonl")
REPORT_MD = Path("eval/runs/clarification_loop.md")
REPORT_JSON = Path("eval/runs/clarification_loop.json")
QUESTIONS_JSONL = Path("eval/runs/clarification_questions.jsonl")

# Phrases a generic "just send more detail" question uses. A question that
# matches one of these is flagged for review -- the DoD wants field-specific
# questions.
_GENERIC_RE = re.compile(
    r"mais detalhes|mais informa\w+|poderia detalhar|fornecer detalhes|"
    r"mais dados|forne\wa mais|envie mais",
    re.IGNORECASE,
)


class _NoopRetriever:
    def retrieve(
        self, question: str, *, k: int, metadata_filter: object | None = None
    ) -> list[str]:
        return []


@dataclass
class _ClaimResult:
    claim_id: str
    missing_fact_type: str
    narrative: str
    generated_questions: list[dict[str, str]] = field(default_factory=list)
    generic_flagged: bool = False
    live_entered_loop: bool = False
    live_rounds: int = 0
    live_exhausted: bool = False
    live_terminated: bool = False
    error: str | None = None


@dataclass
class _Summary:
    n_claims: int
    questions_generated: int
    generic_flagged: int
    min_question_chars: int
    mean_question_chars: float
    live_terminated: int
    live_entered_loop: int
    live_rounds_distribution: dict[str, int]
    error_claim_ids: list[str]
    total_latency_seconds: float


@dataclass
class ClarificationEvalResult:
    """The aggregate summary plus every per-claim result row."""

    meta: dict[str, object]
    summary: _Summary
    results: list[_ClaimResult]

    def to_json(self) -> dict[str, object]:
        """Return the JSON-serialisable ``{meta, summary}`` report body."""
        return {"meta": self.meta, "summary": dataclasses.asdict(self.summary)}


def load_incomplete_claims(path: Path = CLAIMS_PATH) -> list[SyntheticClaim]:
    """The ``insufficient_information`` claims -- the ones with a missing fact."""
    rows = [
        SyntheticClaim.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return [r for r in rows if r.missing_fact_type is not None]


def _evaluate_claim(claim: SyntheticClaim, context: GraphContext) -> _ClaimResult:
    assert claim.missing_fact_type is not None
    tag = claim.missing_fact_type.value
    result = _ClaimResult(
        claim_id=claim.claim_id, missing_fact_type=tag, narrative=claim.narrative
    )
    try:
        state: ClaimState = {
            "claim_id": claim.claim_id,
            "raw_claim_text": claim.narrative,
            "missing_information": [tag],
        }
        update = clarification(state, Runtime(context=context))
        questions = update["clarification_questions"]
        assert isinstance(questions, list)
        result.generated_questions = [
            {"field": q.field, "question": q.question} for q in questions
        ]
        result.generic_flagged = any(_GENERIC_RE.search(q.question) for q in questions)

        compiled = build_claim_graph().compile()
        out = compiled.invoke(
            {"claim_id": claim.claim_id, "raw_claim_text": claim.narrative},
            context=context,
        )
        result.live_terminated = True
        result.live_rounds = int(out.get("clarification_rounds", 0) or 0)
        result.live_entered_loop = result.live_rounds > 0
        result.live_exhausted = bool(out.get("clarification_exhausted", False))
    except Exception as exc:  # noqa: BLE001 -- recorded, the run continues
        result.error = f"{type(exc).__name__}: {exc}"
    return result


def _compute_summary(
    results: list[_ClaimResult], total_latency_seconds: float
) -> _Summary:
    ok = [r for r in results if r.error is None]
    lengths = [len(q["question"]) for r in ok for q in r.generated_questions]
    return _Summary(
        n_claims=len(results),
        questions_generated=sum(len(r.generated_questions) for r in ok),
        generic_flagged=sum(1 for r in ok if r.generic_flagged),
        min_question_chars=min(lengths) if lengths else 0,
        mean_question_chars=round(sum(lengths) / len(lengths), 1) if lengths else 0.0,
        live_terminated=sum(1 for r in ok if r.live_terminated),
        live_entered_loop=sum(1 for r in ok if r.live_entered_loop),
        live_rounds_distribution={
            str(k): v for k, v in sorted(Counter(r.live_rounds for r in ok).items())
        },
        error_claim_ids=[r.claim_id for r in results if r.error is not None],
        total_latency_seconds=round(total_latency_seconds, 2),
    )


def run_clarification_eval(
    model: BaseChatModel | None = None,
) -> ClarificationEvalResult:
    """Run both passes over the incomplete claims and aggregate."""
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
    claims = load_incomplete_claims()
    started = time.monotonic()
    results = [_evaluate_claim(claim, context) for claim in claims]
    elapsed = time.monotonic() - started

    meta: dict[str, object] = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "model": settings.llm_model_fast,
        "provider_order": settings.llm_fast_provider_order,
        "max_clarification_rounds": MAX_CLARIFICATION_ROUNDS,
        "n_incomplete_claims": len(claims),
        "dod_note": (
            "DoD [M4-03] item 5 asks for ≥10 incomplete [M2-04] claims; the "
            f"finalized set has {len(claims)}. Structural termination is proven "
            "in tests/unit/infrastructure/graph/test_claim_graph.py; this run "
            "is the live question-quality check (item 2)."
        ),
    }
    return ClarificationEvalResult(meta, _compute_summary(results, elapsed), results)


def render_markdown(result: ClarificationEvalResult) -> str:
    """Render the ``eval/runs/clarification_loop.md`` body."""
    s = result.summary
    lines = [
        "# Clarification loop -- live measurement ([M4-03])",
        "",
        f"- generated: `{result.meta['generated_at_utc']}`",
        f"- model: `{result.meta['model']}` "
        f"(provider order `{result.meta['provider_order']}`)",
        f"- cap: `MAX_CLARIFICATION_ROUNDS = "
        f"{result.meta['max_clarification_rounds']}`",
        f"- {result.meta['dod_note']}",
        f"- processed {s.n_claims} claims, {len(s.error_claim_ids)} errors "
        f"{s.error_claim_ids or ''}",
        f"- wall clock: {s.total_latency_seconds}s",
        "",
        "## Termination (full compiled graph, real fast model both nodes)",
        "",
        f"- terminated without hanging: **{s.live_terminated}/{s.n_claims}**",
        f"- actually entered the clarification loop live: {s.live_entered_loop}"
        f"/{s.n_claims} (the rest: intake did not re-flag the omitted fact -- see "
        "`docs/INTAKE_EXTRACTION.md` on the pre-retrieval recall frontier)",
        f"- `clarification_rounds` distribution: {s.live_rounds_distribution}",
        "",
        "## Question quality (one forced clarification call per claim)",
        "",
        f"- questions generated: **{s.questions_generated}/{s.n_claims}**",
        f"- flagged as generic phrasing: {s.generic_flagged}/{s.n_claims}",
        f"- question length: min {s.min_question_chars} chars, "
        f"mean {s.mean_question_chars}",
        "",
        "The full per-claim question dump is in "
        "`eval/runs/clarification_questions.jsonl`. A sample:",
        "",
    ]
    for r in result.results:
        if r.error is not None:
            lines.append(
                f"- **{r.claim_id}** (`{r.missing_fact_type}`): ERROR {r.error}"
            )
            continue
        asked = " / ".join(q["question"] for q in r.generated_questions)
        flag = " ⚠ generic" if r.generic_flagged else ""
        lines.append(f"- **{r.claim_id}** (`{r.missing_fact_type}`){flag}: {asked}")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    """Run the measurement and write the reports under ``eval/runs/``."""
    result = run_clarification_eval()

    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(
        json.dumps(result.to_json(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    REPORT_MD.write_text(render_markdown(result), encoding="utf-8")
    with QUESTIONS_JSONL.open("w", encoding="utf-8") as handle:
        for r in result.results:
            handle.write(json.dumps(dataclasses.asdict(r), ensure_ascii=False) + "\n")

    print(render_markdown(result))


if __name__ == "__main__":
    main()
