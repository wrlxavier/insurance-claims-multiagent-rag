"""The rendered-reasoning contract: render/parse are exact inverses [M4-10].

``CompatibilityAssessment.reasoning`` is a string, so [M4-10]'s citation check
recovers the assertions by parsing it. These tests pin the round trip, because
a silent format drift would not fail anything else -- it would just report an
assertion count of zero and a citation coverage of a vacuous 100%.
"""

import pytest

from infrastructure.graph.reasoning_format import (
    NO_ASSERTIONS_REASONING,
    parse_reasoning,
    render_reasoning,
)
from infrastructure.graph.schemas import ReasonedAssertion


def _assertion(statement: str, *clause_ids: str) -> ReasonedAssertion:
    return ReasonedAssertion(statement=statement, clause_ids=list(clause_ids))


@pytest.mark.unit
def test_round_trip_preserves_statements_and_clause_ids() -> None:
    assertions = [
        _assertion("O evento é um roubo.", "16:condicoes-especiais-2/1", "16:5/5.1"),
        _assertion("A cobertura não alcança o veículo.", "16:1"),
    ]
    assert parse_reasoning(render_reasoning(assertions)) == assertions


@pytest.mark.unit
def test_rendered_shape_is_the_documented_one() -> None:
    rendered = render_reasoning([_assertion("Afirmação.", "1:a", "2:b")])
    assert rendered == "1. Afirmação. [cláusulas: 1:a, 2:b]"


@pytest.mark.unit
def test_an_assertion_with_no_clause_id_round_trips_as_empty() -> None:
    # Only reachable on an abstention -- a settled verdict is rejected upstream
    # by `_grounding_errors` -- but the format must not lose the distinction.
    assertions = [_assertion("Nada foi possível fundamentar.")]
    rendered = render_reasoning(assertions)
    assert "[cláusulas: —]" in rendered
    assert parse_reasoning(rendered) == assertions


@pytest.mark.unit
def test_no_assertions_renders_and_parses_back_to_nothing() -> None:
    rendered = render_reasoning([])
    assert rendered == NO_ASSERTIONS_REASONING
    assert parse_reasoning(rendered) == []


@pytest.mark.unit
def test_abstention_prose_parses_to_no_assertions_rather_than_raising() -> None:
    prose = (
        "A avaliação não pôde ser fundamentada nas cláusulas recuperadas após "
        "3 tentativas."
    )
    assert parse_reasoning(prose) == []


@pytest.mark.unit
def test_a_statement_containing_brackets_still_round_trips() -> None:
    assertions = [_assertion("O item [b] do rol não se aplica.", "3:x")]
    assert parse_reasoning(render_reasoning(assertions)) == assertions


@pytest.mark.unit
def test_multiline_reasoning_recovers_every_assertion_in_order() -> None:
    assertions = [_assertion(f"Afirmação {i}.", f"{i}:c") for i in range(1, 6)]
    parsed = parse_reasoning(render_reasoning(assertions))
    assert [a.statement for a in parsed] == [a.statement for a in assertions]
