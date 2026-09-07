"""The claim entity [M5-01].

The free-text loss narrative a policyholder submits, before the graph makes
any structured sense of it. ``raw_text`` is ``ClaimState.raw_claim_text``;
the structured read the intake node produces (``ExtractedEntities``) stays
in ``infrastructure.graph.state`` -- it is graph working state, not a
business entity the DoD names.

``policy_ref`` is the registered product the claim is filed against.
``None`` is the honest default today: intake *extracts* the SUSEP process
from the narrative, so it is not known at submission. [M5-04] makes it a
first-class field of the submission API.

Standard library only -- enforced by
tests/architecture/test_layer_boundaries.py.
"""

from dataclasses import dataclass
from datetime import datetime

from domain.susep_process import SusepProcess


@dataclass(frozen=True)
class Claim:
    """A submitted claim narrative awaiting or carrying an assessment."""

    claim_id: str
    raw_text: str
    submitted_at: datetime
    policy_ref: SusepProcess | None = None

    def __post_init__(self) -> None:
        """Reject an empty id or narrative, or a naive ``submitted_at``."""
        for name in ("claim_id", "raw_text"):
            if not getattr(self, name):
                raise ValueError(f"Claim.{name} must not be empty")
        if self.submitted_at.tzinfo is None or self.submitted_at.utcoffset() is None:
            raise ValueError("Claim.submitted_at must be timezone-aware")
