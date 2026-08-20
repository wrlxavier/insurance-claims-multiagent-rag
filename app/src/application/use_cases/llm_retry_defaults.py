"""Shared default retry policy for transient LLM call failures.

[M1-09]: consolidated after finding the same two values (3 attempts, 5s
apart) independently defined under three different names in
clause_classification.py, boundary_escalation.py and
scripts/validate_parsing_quality_sample.py, with no reason to diverge.
"""

DEFAULT_LLM_RETRY_MAX_ATTEMPTS = 3
DEFAULT_LLM_RETRY_DELAY_SECONDS = 5.0
