"""Deterministic stratified sampling for golden-set-v1's second-reviewer pass.

Selection only -- no model call lives here, matching the same
selection/drafting split ``unanswerable_question_selection.py`` uses for
[M2-05]. Given every ``GoldenQuestion`` row currently in
``data/golden_set/*.jsonl``, grouped by ``question_type``, this module
picks which rows the independent reviewer sees.

Two sampling rules, both fixed here (not tuned after seeing any review
result):

- Every stratum except ``unanswerable`` is sampled at
  [GENERAL_SAMPLE_RATE] (~20%), rounded to the nearest whole question.
- ``unanswerable`` is sampled at the higher [UNANSWERABLE_SAMPLE_RATE]
  (~50%): a wrong ``unanswerable`` label is the one error that silently
  miscalibrates the insufficient-context gate downstream ([M3-07]), so this
  stratum gets more coverage than the rest. The rows drawn beyond its
  ~20% share are tagged ``sample_stratum="unanswerable_topup"`` rather than
  being a separate mechanism -- for an already-blind reviewer, "re-confirm
  this is unanswerable" and "answer this question from the document" are
  the same task.
- The ``cross_document`` stratum's draw is topped up (if needed) to
  guarantee at least one HDI-brand-collision question (the only
  cross_document rows targeting a HDI-branded document,
  [HDI_DOCUMENT_ID]) and at least one ``bundle_section`` question (a
  reference clause whose parsed record carries a non-null
  ``bundle_section``), so neither trap case is missed by chance.

Reproducible via [REVIEW_SAMPLE_SEED] -- a fixed seed, not re-rolled.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

from infrastructure.evaluation.golden_set_schema import GoldenQuestion, QuestionType
from infrastructure.parsing.clause_schema import ParsedClauseRecord

GENERAL_SAMPLE_RATE = 0.20
UNANSWERABLE_SAMPLE_RATE = 0.50
HDI_DOCUMENT_ID = "12"
REVIEW_SAMPLE_SEED = 2207


@dataclass(frozen=True)
class SampledQuestion:
    """One question selected for the independent second-reviewer pass."""

    question_id: str
    question_type: QuestionType
    document_id: str
    sample_stratum: Literal["general_stratified", "unanswerable_topup"]


def _has_bundle_section(
    question: GoldenQuestion, clause_by_id: dict[str, ParsedClauseRecord]
) -> bool:
    """Return True if any of the question's reference clauses is bundled."""
    return any(
        clause_by_id[clause_id].bundle_section
        for clause_id in question.reference_clause_ids
        if clause_id in clause_by_id
    )


def _ensure_cross_document_subsets(
    chosen: list[GoldenQuestion],
    pool: list[GoldenQuestion],
    clause_by_id: dict[str, ParsedClauseRecord],
) -> list[GoldenQuestion]:
    """Top up ``chosen`` so the HDI-collision and bundle_section cases appear.

    Swaps a trailing slot in ``chosen`` for a missing subset's row rather
    than growing the sample size, unless ``chosen`` is empty. Uses a
    decrementing swap index so the two subsets (if both need topping up)
    never clobber the same slot.
    """
    chosen = list(chosen)
    chosen_ids = {row.question_id for row in chosen}
    swap_index = len(chosen) - 1

    subsets = (
        [row for row in pool if row.document_id == HDI_DOCUMENT_ID],
        [row for row in pool if _has_bundle_section(row, clause_by_id)],
    )
    for subset in subsets:
        if not subset or any(row.question_id in chosen_ids for row in subset):
            continue
        replacement = subset[0]
        if swap_index < 0:
            chosen.append(replacement)
        else:
            chosen_ids.discard(chosen[swap_index].question_id)
            chosen[swap_index] = replacement
            swap_index -= 1
        chosen_ids.add(replacement.question_id)
    return chosen


def select_review_sample(
    rows_by_type: dict[QuestionType, list[GoldenQuestion]],
    clause_by_id: dict[str, ParsedClauseRecord],
    *,
    seed: int = REVIEW_SAMPLE_SEED,
) -> list[SampledQuestion]:
    """Return the full stratified sample, sorted by ``question_id``.

    ``rows_by_type`` must cover every [QuestionType] present in the golden
    set; a missing key is treated as an empty stratum.
    """
    rng = random.Random(seed)
    selected: dict[str, SampledQuestion] = {}

    for question_type, rows in rows_by_type.items():
        if question_type is QuestionType.UNANSWERABLE or not rows:
            continue
        pool = list(rows)
        rng.shuffle(pool)
        target = max(round(len(pool) * GENERAL_SAMPLE_RATE), 1)
        chosen = pool[:target]
        if question_type is QuestionType.CROSS_DOCUMENT:
            chosen = _ensure_cross_document_subsets(chosen, pool, clause_by_id)
        for row in chosen:
            selected[row.question_id] = SampledQuestion(
                question_id=row.question_id,
                question_type=question_type,
                document_id=row.document_id,
                sample_stratum="general_stratified",
            )

    unanswerable_rows = rows_by_type.get(QuestionType.UNANSWERABLE, [])
    if unanswerable_rows:
        pool = list(unanswerable_rows)
        rng.shuffle(pool)
        general_target = max(round(len(pool) * GENERAL_SAMPLE_RATE), 1)
        total_target = max(round(len(pool) * UNANSWERABLE_SAMPLE_RATE), general_target)
        for row in pool[:general_target]:
            selected[row.question_id] = SampledQuestion(
                question_id=row.question_id,
                question_type=QuestionType.UNANSWERABLE,
                document_id=row.document_id,
                sample_stratum="general_stratified",
            )
        for row in pool[general_target:total_target]:
            selected[row.question_id] = SampledQuestion(
                question_id=row.question_id,
                question_type=QuestionType.UNANSWERABLE,
                document_id=row.document_id,
                sample_stratum="unanswerable_topup",
            )

    return sorted(selected.values(), key=lambda sampled: sampled.question_id)
