"""The adversarial prompt-injection fixture schema ([M5-08]).

Fixes the format every row of ``data/adversarial_injection/fixtures.jsonl``
must follow, mirroring ``golden_set_schema.py`` / ``synthetic_claims_schema.py``'s
pattern: a flat, validated Pydantic row that fails loudly on a malformed
fixture rather than accepting it silently.

Unlike those two, these rows are hand-authored, not produced by the
deterministic-selection + LLM-phrasing + human-review pipeline
(``data/adversarial_injection/README.md`` explains why) -- but the same
"validate, don't coerce" discipline applies to the fixture file itself.

Kept independent of ``infrastructure.graph``: ``entities`` and ``citations``
here are plain nested models, not ``state.ExtractedEntities`` /
``state.Citation``. ``scripts/eval_prompt_injection.py`` -- which already
depends on the graph package to invoke the compatibility node -- converts
between the two; this module stays usable without a LangGraph import.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "v1"

FixtureKind = Literal["clause_injection", "claim_injection"]
FixtureVerdict = Literal["compatible", "incompatible", "insufficient_information"]


class FixtureEntities(BaseModel):
    """The subset of ``ExtractedEntities`` fields a fixture may set."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_type: str | None = None
    event_date: str | None = None
    description: str | None = None
    estimated_amount: float | None = None
    vehicle_info: str | None = None
    susep_process: str | None = None
    product_line: str | None = None


class FixtureCitation(BaseModel):
    """The subset of ``Citation`` fields a fixture sets directly.

    No retrieval runs for these fixtures -- the citation list a real
    retrieval call would have produced is supplied here instead, so the
    excerpt's injected instruction is exactly what the compatibility node
    sees.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    clause_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    susep_process: str = Field(min_length=1)
    clause_type: str
    relevance_score: float = Field(ge=0.0)
    excerpt: str = Field(min_length=1)


class AdversarialInjectionFixture(BaseModel):
    """One adversarial probe: a poisoned clause, or a clean/injected claim pair."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str
    fixture_id: str = Field(min_length=1)
    kind: FixtureKind
    entities: FixtureEntities
    citations: list[FixtureCitation] = Field(min_length=1)
    claim_narrative: str | None = None
    claim_narrative_clean: str | None = None
    claim_narrative_injected: str | None = None
    expected_verdict: FixtureVerdict
    notes: str = ""

    @model_validator(mode="after")
    def _narrative_matches_kind(self) -> "AdversarialInjectionFixture":
        if self.kind == "clause_injection":
            if not self.claim_narrative:
                raise ValueError("clause_injection fixture needs claim_narrative")
            if self.claim_narrative_clean or self.claim_narrative_injected:
                raise ValueError(
                    "clause_injection fixture must not set the claim_injection "
                    "narrative pair"
                )
        else:
            if not self.claim_narrative_clean or not self.claim_narrative_injected:
                raise ValueError(
                    "claim_injection fixture needs both claim_narrative_clean "
                    "and claim_narrative_injected"
                )
            if self.claim_narrative:
                raise ValueError("claim_injection fixture must not set claim_narrative")
        return self
