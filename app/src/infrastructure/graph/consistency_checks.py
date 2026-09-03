"""The deterministic half of the consistency node -- plain Python, no LLM ([M4-06]).

[M4-06] splits the consistency node's work along one line. Whatever can be
decided by parsing a value, comparing a number to a constant, or testing set
membership is arithmetic: it runs here, unconditionally, with no model in the
loop. Whatever needs reading comprehension -- does the narrative hold together,
does the free description match the labelled event type -- is judgement, and
lives in ``nodes/consistency.py`` behind one fast-model call. The rule of thumb:
*string / number / set equality is Python; "do these two prose fields disagree
in meaning" is the model's.*

Every function here returns ``state.ConsistencySignal`` rows with
``source="deterministic"``. A signal is a flag for a human reviewer, never a
verdict -- this module decides nothing.

**This is not a fraud detector, and the project does not claim it is.** The
checks catch a date typed in the future, an amount that is negative or absurd, a
field intake contradicted elsewhere in its own extraction, and a product line
whose own registered definition rules out the event that was described. None of
that is fraud detection: the data carries no fraud labels and a handful of range
checks is not a method for it. See ``docs/SCOPE.md``.

What is deliberately *not* checked here: whether the event date falls inside the
policy period. The corpus is registered product conditions, not contracts -- it
carries no vigência, insured amount or deductible (``docs/SCOPE.md``) -- so there
is no period to compare a date against. Intake's ``data_evento_vigencia``
missing-information tag records that gap instead.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Literal

from infrastructure.graph.state import ConsistencySignal, ExtractedEntities

_Severity = Literal["info", "attention"]

# Stable signal identifiers, so ``scripts/eval_consistency.py`` can aggregate
# signal counts by check across the synthetic set.
CHECK_DATE_IN_FUTURE = "date_in_future"
CHECK_EVENT_DATE_FAR_PAST = "event_date_far_past"
CHECK_AMOUNT_NON_POSITIVE = "amount_non_positive"
CHECK_AMOUNT_IMPLAUSIBLY_LOW = "amount_implausibly_low"
CHECK_AMOUNT_IMPLAUSIBLY_HIGH = "amount_implausibly_high"
CHECK_FIELD_CONTRADICTS_MISSING_TAG = "field_contradicts_missing_info_tag"
CHECK_PRODUCT_LINE_CONTRADICTS_EVENT = "product_line_contradicts_event"

DETERMINISTIC_CHECK_NAMES: tuple[str, ...] = (
    CHECK_DATE_IN_FUTURE,
    CHECK_EVENT_DATE_FAR_PAST,
    CHECK_AMOUNT_NON_POSITIVE,
    CHECK_AMOUNT_IMPLAUSIBLY_LOW,
    CHECK_AMOUNT_IMPLAUSIBLY_HIGH,
    CHECK_FIELD_CONTRADICTS_MISSING_TAG,
    CHECK_PRODUCT_LINE_CONTRADICTS_EVENT,
)

# Amount sanity rails, in BRL. Arbitrary by design: they catch a sign error or
# an order-of-magnitude typo on a passenger-vehicle claim, nothing subtler.
# Not fraud thresholds.
_MIN_PLAUSIBLE_AMOUNT = 100.0
_MAX_PLAUSIBLE_AMOUNT = 2_000_000.0

# How far before "now" an absolute event date may fall before it is worth a
# reviewer's glance -- a claim reported many years after the event. Expressed in
# days to sidestep the 29-Feb edge of subtracting years from a ``date``.
_STALE_EVENT_DAYS = int(7 * 365.25)

# Accepted absolute-date formats for ``event_date``. Anything else -- a relative
# phrase ("faz umas duas semanas"), a malformed string -- yields no signal: a
# vague date is not a contradiction, it is a gap intake already tags.
_DATE_FORMATS = ("%Y-%m-%d", "%d/%m/%Y", "%d/%m/%y")

# Token vocabularies for the product-line/event-type contradiction table. The
# peril tokens are adapted from
# ``scripts/synthetic_claims_selection._PERIL_HINT_PATTERN``; the graph package
# cannot import from ``scripts/`` (layering), the same reason
# ``schemas.MissingInfoTag`` is redeclared rather than imported.
_EXTERNAL_CAUSE_RE = re.compile(
    r"colis[ãa]o|abalroament|batid[ao]|\bbati\b|bateu|inc[êe]ndio|\bfogo\b|"
    r"roubo|furto|assalto|granizo|alagament|enchente|capotament|vandalismo|"
    r"acidente",
    re.IGNORECASE,
)
_VEHICLE_INDEMNITY_RE = re.compile(
    r"perda total|indeniza[çc][ãa]o (do|integral|de mercado do) ve[íi]culo|"
    r"valor de mercado do ve[íi]culo|valor do ve[íi]culo",
    re.IGNORECASE,
)

# A negation immediately before a peril token flips its meaning -- a GAR.EST
# claimant routinely writes "não teve colisão" or "nada a ver com batida" to
# establish that the failure was self-caused. The check scans a short window
# before the matched token and stays silent when it is negated.
_NEGATION_RE = re.compile(
    r"\b(n[ãa]o|nunca|jamais|sem|descartad[ao])\b|nada a ver com",
    re.IGNORECASE,
)
_NEGATION_WINDOW_CHARS = 45

# product_line -> the pattern whose (un-negated) match contradicts that line.
# Only the two lines whose registered definition carries an *absolute*
# constraint: GAR.EST is a self-caused part failure "with no crash or external
# cause" (data/README.md / prompts.intake.PRODUCT_LINE_GUIDE), and ASSIST is "a
# service ... rather than indemnification for vehicle damage". CASCO vs. RCF-A
# turns on *who* was damaged -- a distinction a terse extracted description
# renders unreliably (measured: it misfired on coherent CASCO claims in the
# first eval run), so those lines are deliberately not in the table. CARTA VERDE
# is not contradicted by any single token. The check fires only on a positive,
# un-negated collision; it never proposes a different line and stays silent when
# product_line is None.
_CONTRADICTION_BY_LINE: dict[str, re.Pattern[str]] = {
    "GAR.EST": _EXTERNAL_CAUSE_RE,
    "ASSIST": _VEHICLE_INDEMNITY_RE,
}

# field on ExtractedEntities -> the missing_information tag that contradicts it
# being populated. Intake disagreeing with its own extraction.
_TAG_FOR_FIELD: dict[str, str] = {
    "estimated_amount": "valor_franquia_limite",
    "event_date": "data_evento_vigencia",
}


def _signal(check: str, severity: _Severity, detail: str) -> ConsistencySignal:
    """Build a deterministic ``ConsistencySignal`` (``source`` is fixed here)."""
    return ConsistencySignal(
        check=check, severity=severity, detail=detail, source="deterministic"
    )


def _parse_absolute_date(raw: str) -> date | None:
    """Parse ``raw`` as an absolute date in one of ``_DATE_FORMATS``, else ``None``."""
    stripped = raw.strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(stripped, fmt).date()
        except ValueError:
            continue
    return None


def check_date_coherence(
    entities: ExtractedEntities, *, now: datetime
) -> list[ConsistencySignal]:
    """Flag an absolute ``event_date`` that is in the future or long past.

    ``now`` is injected (the node passes ``datetime.now(UTC)``) so the check is
    a pure function under test. A relative phrase or an unparseable string
    produces no signal -- this is not the place to judge a vague date.
    """
    raw = entities.event_date
    if not raw:
        return []
    parsed = _parse_absolute_date(raw)
    if parsed is None:
        return []
    today = now.date()
    if parsed > today:
        return [
            _signal(
                CHECK_DATE_IN_FUTURE,
                "attention",
                f"A data informada do evento ({parsed.isoformat()}) é "
                "posterior à data de hoje.",
            )
        ]
    if (today - parsed).days > _STALE_EVENT_DAYS:
        return [
            _signal(
                CHECK_EVENT_DATE_FAR_PAST,
                "info",
                f"A data informada do evento ({parsed.isoformat()}) é de mais "
                "de sete anos atrás.",
            )
        ]
    return []


def check_amount_plausibility(
    entities: ExtractedEntities,
) -> list[ConsistencySignal]:
    """Flag an ``estimated_amount`` outside crude BRL sanity bands.

    The bands catch a sign error or an order-of-magnitude typo, nothing more --
    they are not fraud thresholds. Silent when no amount was extracted.
    """
    amount = entities.estimated_amount
    if amount is None:
        return []
    if amount <= 0:
        return [
            _signal(
                CHECK_AMOUNT_NON_POSITIVE,
                "attention",
                f"O valor estimado do prejuízo ({amount}) não é positivo.",
            )
        ]
    if amount < _MIN_PLAUSIBLE_AMOUNT:
        return [
            _signal(
                CHECK_AMOUNT_IMPLAUSIBLY_LOW,
                "info",
                f"O valor estimado do prejuízo (R$ {amount:.2f}) está abaixo de "
                "qualquer franquia usual.",
            )
        ]
    if amount > _MAX_PLAUSIBLE_AMOUNT:
        return [
            _signal(
                CHECK_AMOUNT_IMPLAUSIBLY_HIGH,
                "attention",
                f"O valor estimado do prejuízo (R$ {amount:,.2f}) está muito "
                "acima do usual para um veículo de passeio; possível erro de "
                "digitação.",
            )
        ]
    return []


def check_internal_contradictions(
    entities: ExtractedEntities, missing_information: list[str]
) -> list[ConsistencySignal]:
    """Flag a field intake populated while also tagging it as missing.

    Only the crisp, equality-checkable cases: an ``estimated_amount`` with
    ``valor_franquia_limite`` still in ``missing_information``, or an
    ``event_date`` with ``data_evento_vigencia`` still there. "These two prose
    fields disagree" is the model's job, not this one's.
    """
    tags = set(missing_information)
    signals: list[ConsistencySignal] = []
    for field_name, tag in _TAG_FOR_FIELD.items():
        if getattr(entities, field_name) is not None and tag in tags:
            signals.append(
                _signal(
                    CHECK_FIELD_CONTRADICTS_MISSING_TAG,
                    "info",
                    f"O campo '{field_name}' foi extraído, mas '{tag}' consta "
                    "como informação ausente.",
                )
            )
    return signals


def _is_negated(text: str, match: re.Match[str]) -> bool:
    """Whether a negation word sits just before ``match`` in ``text``."""
    window = text[max(0, match.start() - _NEGATION_WINDOW_CHARS) : match.start()]
    return _NEGATION_RE.search(window) is not None


def check_product_line_event_type_mismatch(
    entities: ExtractedEntities,
) -> list[ConsistencySignal]:
    """Flag a classified product line whose own definition rules out the event.

    A closed contradiction table (``_CONTRADICTION_BY_LINE``), taken from the two
    registered products with an absolute constraint -- GAR.EST (no external
    cause) and ASSIST (a service, not indemnification). It fires only on a
    positive, un-negated token in ``event_type`` / ``description`` and never
    proposes an alternative line. Silent when ``product_line`` is ``None`` or a
    line not in the table. The deliberate overlap with intake's own
    classification guidance is the point: a hit means intake's classification
    disagrees with the event text intake itself extracted.
    """
    line = entities.product_line
    if line is None or line not in _CONTRADICTION_BY_LINE:
        return []
    text = " ".join(
        part for part in (entities.event_type, entities.description) if part
    ).strip()
    if not text:
        return []
    match = _CONTRADICTION_BY_LINE[line].search(text)
    if match is None or _is_negated(text, match):
        return []
    return [
        _signal(
            CHECK_PRODUCT_LINE_CONTRADICTS_EVENT,
            "attention",
            f"O ramo classificado ({line}) é contrariado pelo evento descrito "
            f"(trecho: {match.group(0)!r}).",
        )
    ]


def run_deterministic_checks(
    entities: ExtractedEntities | None,
    missing_information: list[str],
    *,
    now: datetime,
) -> list[ConsistencySignal]:
    """Run every deterministic check and return the concatenated signals.

    ``entities is None`` (intake extracted nothing structured) yields ``[]`` --
    there is nothing to check arithmetically. Every returned signal has
    ``source="deterministic"``.
    """
    if entities is None:
        return []
    return [
        *check_date_coherence(entities, now=now),
        *check_amount_plausibility(entities),
        *check_internal_contradictions(entities, missing_information),
        *check_product_line_event_type_mismatch(entities),
    ]
