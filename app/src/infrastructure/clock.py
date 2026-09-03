"""The real ``Clock`` -- [M5-04].

``application.ports.clock.Clock`` asks for one method: the current instant as a
timezone-aware UTC ``datetime``. The port docstring assigns the concrete
implementation to the composition root; this is it, a one-line
``datetime.now(UTC)`` wrapper. Kept as a class so it satisfies the ``Clock``
protocol structurally and a test can still swap in a ``FixedClock``.
"""

from datetime import UTC, datetime


class SystemClock:
    """A ``Clock`` backed by the wall clock."""

    def now(self) -> datetime:
        """Return the current instant as a timezone-aware UTC ``datetime``."""
        return datetime.now(UTC)
