"""Tests for the rules loader."""

from pathlib import Path

from domain.clause_classification import ClauseType
from infrastructure.parsing.rules_loader import load_classification_rules


def test_load_classification_rules(tmp_path: Path) -> None:
    # Create a temporary CSV
    csv_file = tmp_path / "rules.csv"
    csv_file.write_text(
        "priority,heading_pattern,clause_type\n"
        "10,^riscos excluídos$,exclusion\n"
        "20,^exclusões$,exclusion\n"
        "10,^cobertura básica$,coverage\n",
        encoding="utf-8",
    )

    rules = load_classification_rules(csv_file)

    # Should be sorted by priority descending
    assert len(rules) == 3

    pattern_1, type_1 = rules[0]
    assert pattern_1.pattern == "^exclusoes$"
    assert type_1 == ClauseType.EXCLUSION

    pattern_2, type_2 = rules[1]
    # The order of the two priority 10 rules is stable (same as input)
    assert pattern_2.pattern == "^riscos excluidos$"
    assert type_2 == ClauseType.EXCLUSION

    pattern_3, type_3 = rules[2]
    assert pattern_3.pattern == "^cobertura basica$"
    assert type_3 == ClauseType.COVERAGE
