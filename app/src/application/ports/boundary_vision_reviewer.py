"""Port for the vision-LLM boundary review ([M1-04d])."""

from typing import Protocol

from domain.boundary_escalation import BoundaryReview


class BoundaryVisionReviewerPort(Protocol):
    """Interface for asking a vision-capable model to review a clause boundary."""

    def review(
        self,
        *,
        clause_title: str,
        claimed_page_start: int,
        claimed_page_end: int,
        page_images: tuple[bytes, ...],
    ) -> BoundaryReview:
        """Review a deterministic pass's claimed boundary against page images.

        Args:
            clause_title: The clause's title, as recorded by the
                deterministic pass.
            claimed_page_start: The deterministic pass's recorded page_start.
            claimed_page_end: The deterministic pass's recorded page_end.
            page_images: PNG-encoded images of the claimed range plus margin,
                in page order.

        Returns:
            The model's judgment: confirmation, a corrected page range, or a
            suggested sub-clause split.
        """
        ...
