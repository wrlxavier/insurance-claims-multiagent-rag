"""The independent second-reviewer schema for golden-set-v1.

A deliberately separate model from [infrastructure.evaluation.
golden_set_schema.GoldenQuestion] rather than optional fields bolted onto
it: both ``scripts/eval_retrieval.py`` and ``scripts/validate_golden_set.py``
glob ``data/golden_set/*.jsonl`` non-recursively and parse every row as a
``GoldenQuestion`` -- a review row living in that same top-level directory
would either fail that parse or get double-counted as a second question.
Review rows instead live under ``data/golden_set/review/`` (see
``scripts/review_golden_set_sample.py``), joined back onto the original
question by ``question_id`` when a report needs both. This also keeps the
golden set itself frozen: a reviewed question's original row is never
edited in place, whatever the reviewer's own answer turns out to be.

See ``docs/EVALUATION.md``'s "Independent second-reviewer pass" section for
the sampling frame, the review-packet definition, and the adjudication rule
this schema's ``divergence_note``/``adjudication`` fields record.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from infrastructure.evaluation.golden_set_schema import ExpectedVerdict, QuestionType

SCHEMA_VERSION = "review-v1"


class GoldenQuestionReview(BaseModel):
    """One sampled question's independent-review outcome.

    ``agreement`` is ``True`` only when both ``clause_ids_exact_match`` is
    ``True`` and ``verdict_match`` is either ``True`` or ``None`` (no verdict
    dimension applies to this question_type). ``divergence_note`` and
    ``adjudication`` are only ever populated when ``agreement`` is
    ``False`` -- there is nothing to adjudicate on a match.
    """

    schema_version: str
    question_id: str
    question_type: QuestionType
    sample_stratum: Literal["general_stratified", "unanswerable_topup"]
    reviewer_id: str
    reviewed_at: str
    reviewer_reference_clause_ids: list[str]
    reviewer_verdict: ExpectedVerdict | None
    reviewer_rationale: str
    clause_ids_exact_match: bool
    clause_ids_jaccard: float
    verdict_match: bool | None
    agreement: bool
    divergence_note: str | None = None
    adjudication: str | None = None
