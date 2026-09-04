"""The benign-clause fixture schema -- [M5-08 Appendix].

Fixes the format every row of
``data/adversarial_injection/benign_imperative_clauses.jsonl`` must follow --
the true-negative counterpart to ``adversarial_injection_schema.py``'s
adversarial probes. Same "validate, don't coerce" discipline: a malformed row
fails loudly rather than being accepted silently.

These rows are hand-picked, verbatim excerpts from the real parsed corpus
(``build/parsed_clauses.jsonl``), not synthetic and not adversarial -- the
Appendix's own false-positive-rate benchmark needs real imperative SUSEP
clause language the classifier has never been trained against, and
``document_id``/``clause_id``/``susep_process``/``insurer`` are kept so every
row is traceable back to its source PDF.
"""

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "v1"


class BenignClauseFixture(BaseModel):
    """One real, non-adversarial imperative clause excerpt."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str
    fixture_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    clause_id: str = Field(min_length=1)
    clause_type: str
    susep_process: str = Field(min_length=1)
    insurer: str = Field(min_length=1)
    excerpt: str = Field(min_length=1)
    notes: str = ""
