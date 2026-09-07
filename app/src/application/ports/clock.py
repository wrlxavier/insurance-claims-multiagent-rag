"""Port for reading the current time [M5-02].

The use cases stamp a ``Claim.submitted_at`` and a ``HumanDecision.decided_at``,
both of which the domain requires to be timezone-aware. Taking the clock as a
port rather than calling ``datetime.now`` directly is what makes those
timestamps assertable in a unit test (a ``FixedClock``) and keeps the use cases
free of a hidden dependency on wall-clock time -- the same reason
``infrastructure.graph.consistency_checks`` already takes ``now=`` as a
parameter.

The real implementation (a one-line ``datetime.now(UTC)`` wrapper) is the
composition root's to provide -- [M5-04].
"""

from datetime import datetime
from typing import Protocol


class Clock(Protocol):
    """A source of the current instant."""

    def now(self) -> datetime:
        """Return the current instant as a timezone-aware UTC ``datetime``."""
        ...
