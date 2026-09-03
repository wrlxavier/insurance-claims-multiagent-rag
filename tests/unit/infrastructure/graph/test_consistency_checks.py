"""Unit tests for the deterministic consistency checks ([M4-06]).

Every check in isolation, with no LLM anywhere in the module -- the DoD's
"unit-test every deterministic check in isolation" item. ``now`` is passed
explicitly so each test is a pure function call.
"""

from datetime import UTC, datetime

import pytest

from infrastructure.graph import consistency_checks
from infrastructure.graph.consistency_checks import (
    CHECK_AMOUNT_IMPLAUSIBLY_HIGH,
    CHECK_AMOUNT_IMPLAUSIBLY_LOW,
    CHECK_AMOUNT_NON_POSITIVE,
    CHECK_DATE_IN_FUTURE,
    CHECK_EVENT_DATE_FAR_PAST,
    CHECK_FIELD_CONTRADICTS_MISSING_TAG,
    CHECK_PRODUCT_LINE_CONTRADICTS_EVENT,
    check_amount_plausibility,
    check_date_coherence,
    check_internal_contradictions,
    check_product_line_event_type_mismatch,
    run_deterministic_checks,
)
from infrastructure.graph.state import ExtractedEntities

_NOW = datetime(2026, 9, 1, tzinfo=UTC)


def _entities(**kwargs: object) -> ExtractedEntities:
    return ExtractedEntities(**kwargs)


# --- date coherence ---------------------------------------------------------


@pytest.mark.unit
def test_date_in_the_future_is_flagged_attention() -> None:
    signals = check_date_coherence(_entities(event_date="2027-01-15"), now=_NOW)
    assert [s.check for s in signals] == [CHECK_DATE_IN_FUTURE]
    assert signals[0].severity == "attention"
    assert signals[0].source == "deterministic"


@pytest.mark.unit
def test_date_within_a_normal_window_produces_no_signal() -> None:
    assert check_date_coherence(_entities(event_date="2026-08-20"), now=_NOW) == []
    assert check_date_coherence(_entities(event_date="15/08/2026"), now=_NOW) == []


@pytest.mark.unit
def test_date_far_in_the_past_is_flagged_info() -> None:
    signals = check_date_coherence(_entities(event_date="2015-06-01"), now=_NOW)
    assert [s.check for s in signals] == [CHECK_EVENT_DATE_FAR_PAST]
    assert signals[0].severity == "info"


@pytest.mark.unit
def test_relative_or_unparseable_date_produces_no_signal() -> None:
    assert (
        check_date_coherence(_entities(event_date="faz duas semanas"), now=_NOW) == []
    )
    assert check_date_coherence(_entities(event_date="31/02/2025"), now=_NOW) == []
    assert check_date_coherence(_entities(event_date=None), now=_NOW) == []


# --- amount plausibility ---------------------------------------------------


@pytest.mark.unit
def test_amount_none_produces_no_signal() -> None:
    assert check_amount_plausibility(_entities(estimated_amount=None)) == []


@pytest.mark.unit
@pytest.mark.parametrize("amount", [0.0, -50.0])
def test_non_positive_amount_is_flagged_attention(amount: float) -> None:
    signals = check_amount_plausibility(_entities(estimated_amount=amount))
    assert [s.check for s in signals] == [CHECK_AMOUNT_NON_POSITIVE]
    assert signals[0].severity == "attention"


@pytest.mark.unit
def test_implausibly_low_amount_is_flagged_info() -> None:
    signals = check_amount_plausibility(_entities(estimated_amount=42.0))
    assert [s.check for s in signals] == [CHECK_AMOUNT_IMPLAUSIBLY_LOW]
    assert signals[0].severity == "info"


@pytest.mark.unit
def test_implausibly_high_amount_is_flagged_attention() -> None:
    signals = check_amount_plausibility(_entities(estimated_amount=9_000_000.0))
    assert [s.check for s in signals] == [CHECK_AMOUNT_IMPLAUSIBLY_HIGH]
    assert signals[0].severity == "attention"


@pytest.mark.unit
def test_ordinary_amount_produces_no_signal() -> None:
    assert check_amount_plausibility(_entities(estimated_amount=12_500.0)) == []


# --- internal contradictions ---------------------------------------------


@pytest.mark.unit
def test_amount_present_but_tagged_missing_is_flagged_info() -> None:
    signals = check_internal_contradictions(
        _entities(estimated_amount=3_000.0), ["valor_franquia_limite"]
    )
    assert [s.check for s in signals] == [CHECK_FIELD_CONTRADICTS_MISSING_TAG]
    assert signals[0].severity == "info"
    assert "estimated_amount" in signals[0].detail


@pytest.mark.unit
def test_date_present_but_tagged_missing_is_flagged_info() -> None:
    signals = check_internal_contradictions(
        _entities(event_date="2026-08-01"), ["data_evento_vigencia"]
    )
    assert [s.check for s in signals] == [CHECK_FIELD_CONTRADICTS_MISSING_TAG]


@pytest.mark.unit
def test_no_contradiction_when_field_and_tag_do_not_collide() -> None:
    assert (
        check_internal_contradictions(
            _entities(estimated_amount=3_000.0), ["data_evento_vigencia"]
        )
        == []
    )
    assert check_internal_contradictions(_entities(event_date="2026-08-01"), []) == []


# --- product-line / event-type contradiction table ----------------------


@pytest.mark.unit
def test_garest_with_an_external_cause_is_flagged_attention() -> None:
    signals = check_product_line_event_type_mismatch(
        _entities(
            product_line="GAR.EST",
            event_type="colisão",
            description="bati o carro no poste",
        )
    )
    assert [s.check for s in signals] == [CHECK_PRODUCT_LINE_CONTRADICTS_EVENT]
    assert signals[0].severity == "attention"
    assert "GAR.EST" in signals[0].detail


@pytest.mark.unit
def test_garest_with_a_negated_external_cause_is_not_flagged() -> None:
    # a GAR.EST claimant routinely writes "sem colisão" / "nada a ver com
    # batida" to establish the failure was self-caused -- the negation guard
    # must suppress those (measured false positives in the first eval run).
    for description in (
        "defeito interno no módulo, sem colisão ou causa externa",
        "quebra mecânica interna, nada a ver com batida ou mau uso",
        "pane no câmbio; não houve colisão nem impacto",
    ):
        assert (
            check_product_line_event_type_mismatch(
                _entities(product_line="GAR.EST", description=description)
            )
            == []
        )


@pytest.mark.unit
def test_assist_with_vehicle_indemnification_language_is_flagged() -> None:
    signals = check_product_line_event_type_mismatch(
        _entities(
            product_line="ASSIST",
            description="quero a indenização integral do veículo, foi perda total",
        )
    )
    assert [s.check for s in signals] == [CHECK_PRODUCT_LINE_CONTRADICTS_EVENT]


@pytest.mark.unit
def test_casco_and_rcf_a_are_not_in_the_contradiction_table() -> None:
    # CASCO vs RCF-A turns on *who* was damaged -- a terse extracted
    # description renders that unreliably, so neither line is checked.
    assert (
        check_product_line_event_type_mismatch(
            _entities(product_line="CASCO", description="bati no carro do terceiro")
        )
        == []
    )
    assert (
        check_product_line_event_type_mismatch(
            _entities(
                product_line="RCF-A", description="quebrou o para-brisa do meu carro"
            )
        )
        == []
    )


@pytest.mark.unit
def test_product_line_none_or_carta_verde_is_never_flagged() -> None:
    assert (
        check_product_line_event_type_mismatch(
            _entities(product_line=None, description="colisão com incêndio")
        )
        == []
    )
    assert (
        check_product_line_event_type_mismatch(
            _entities(product_line="CARTA VERDE", description="colisão na fronteira")
        )
        == []
    )


@pytest.mark.unit
def test_product_line_set_but_no_event_text_is_not_flagged() -> None:
    assert (
        check_product_line_event_type_mismatch(_entities(product_line="GAR.EST")) == []
    )


# --- run_deterministic_checks -------------------------------------------


@pytest.mark.unit
def test_run_deterministic_checks_concatenates_and_tags_source() -> None:
    entities = _entities(
        event_date="2099-01-01",
        estimated_amount=-1.0,
        product_line="GAR.EST",
        event_type="colisão",
    )
    signals = run_deterministic_checks(entities, [], now=_NOW)
    checks = {s.check for s in signals}
    assert checks == {
        CHECK_DATE_IN_FUTURE,
        CHECK_AMOUNT_NON_POSITIVE,
        CHECK_PRODUCT_LINE_CONTRADICTS_EVENT,
    }
    assert all(s.source == "deterministic" for s in signals)


@pytest.mark.unit
def test_run_deterministic_checks_returns_empty_for_no_entities() -> None:
    assert run_deterministic_checks(None, ["data_evento_vigencia"], now=_NOW) == []


@pytest.mark.unit
def test_a_coherent_claim_produces_no_deterministic_signals() -> None:
    entities = _entities(
        event_date="18/08/2026",
        estimated_amount=7_500.0,
        product_line="CASCO",
        event_type="colisão",
        description="bati meu carro na traseira de outro no semáforo",
    )
    assert run_deterministic_checks(entities, [], now=_NOW) == []


@pytest.mark.unit
def test_module_docstring_states_it_is_not_a_fraud_detector() -> None:
    assert consistency_checks.__doc__ is not None
    assert "not a fraud detector" in consistency_checks.__doc__


@pytest.mark.unit
def test_every_deterministic_signal_name_is_listed() -> None:
    assert set(consistency_checks.DETERMINISTIC_CHECK_NAMES) == {
        CHECK_DATE_IN_FUTURE,
        CHECK_EVENT_DATE_FAR_PAST,
        CHECK_AMOUNT_NON_POSITIVE,
        CHECK_AMOUNT_IMPLAUSIBLY_LOW,
        CHECK_AMOUNT_IMPLAUSIBLY_HIGH,
        CHECK_FIELD_CONTRADICTS_MISSING_TAG,
        CHECK_PRODUCT_LINE_CONTRADICTS_EVENT,
    }
