"""CSV loader for deterministic classification rules."""

import csv
import re
from pathlib import Path

from application.use_cases.clause_classification import normalize_heading
from domain.clause_classification import ClauseType


def load_classification_rules(
    csv_path: Path,
) -> list[tuple[re.Pattern[str], ClauseType]]:
    """Load, sort by priority, normalize, and compile regex rules from CSV.

    The CSV is expected to have headers: priority,heading_pattern,clause_type.
    """
    rules_data = []

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            priority = int(row["priority"])
            pattern_str = row["heading_pattern"]
            clause_type_str = row["clause_type"].upper()

            # Normalize the literal parts of the regex pattern while keeping anchors.
            # E.g. "^riscos excluídos$" -> "^riscos excluidos$"
            # To do this safely for basic regexes used here, we apply the same
            # normalization.
            # Caution: If patterns contain complex regex char classes (like \w),
            # normalization might break them.
            # We assume simple substring or anchor patterns here as per M1-05.
            normalized_pattern_str = normalize_heading(pattern_str)

            clause_type = ClauseType[clause_type_str]
            rules_data.append((priority, normalized_pattern_str, clause_type))

    # Sort descending by priority (higher priority runs first)
    rules_data.sort(key=lambda x: x[0], reverse=True)

    compiled_rules = []
    for _, pattern_str, clause_type in rules_data:
        compiled_rules.append((re.compile(pattern_str), clause_type))

    return compiled_rules
