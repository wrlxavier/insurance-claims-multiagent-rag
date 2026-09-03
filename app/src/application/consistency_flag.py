"""The consistency-flag DTO the application layer speaks in [M5-02].

The application twin of ``infrastructure.graph.state.ConsistencySignal``: one
thing the consistency node flagged for a reviewer's attention. The domain layer
has no equivalent -- ``docs/DOMAIN.md`` defers persisting consistency signals to
[M5-03], and no M5-01 invariant touches them -- so the shape lives here, framed
in the same four fields the graph uses, for ``OrchestratorResult`` and
``AssessmentRecord`` to carry.

Standard library and typing only -- the application layer never imports
Pydantic or ``infrastructure`` (enforced by
tests/architecture/test_layer_boundaries.py).
"""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ConsistencyFlag:
    """One attention point raised while assessing a claim -- never a verdict."""

    check: str
    severity: Literal["info", "attention"]
    detail: str
    source: Literal["deterministic", "llm"]

    def __post_init__(self) -> None:
        """Reject an empty check name or an out-of-vocabulary severity/source."""
        if not self.check:
            raise ValueError("ConsistencyFlag.check must not be empty")
        if self.severity not in ("info", "attention"):
            raise ValueError(
                f"ConsistencyFlag.severity must be 'info' or 'attention', "
                f"got {self.severity!r}"
            )
        if self.source not in ("deterministic", "llm"):
            raise ValueError(
                f"ConsistencyFlag.source must be 'deterministic' or 'llm', "
                f"got {self.source!r}"
            )
