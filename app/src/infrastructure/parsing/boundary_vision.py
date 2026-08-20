"""Vision-LLM boundary review for [M1-04d], and its rasterization port.

``LangchainBoundaryVisionReviewer`` sends a flagged clause's claimed
boundary plus the rasterized page images to a vision-capable model via the
existing [infrastructure.config.llm_client_factory.build_chat_model]
factory -- no new LLM client integration. ``PyMuPdfPageRasterizer`` reuses
the exact PyMuPDF rasterization pattern already used for OCR (see
[infrastructure.parsing.ocr.TesseractOcrExtractor]) and for [M1-08b]'s
``scripts/validate_parsing_quality_sample.py``, with an idempotent disk
cache so a rerun never re-rasterizes an already-cached page.

Nothing outside this module (and its sibling parsing modules) imports
``fitz``/``PIL`` directly -- same convention as [infrastructure.parsing.ocr].
"""

import base64
import time
from pathlib import Path
from typing import cast

import fitz
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from application.ports.boundary_vision_reviewer import BoundaryVisionReviewerPort
from application.ports.page_rasterizer import PageRasterizerPort
from domain.boundary_escalation import BoundaryReview

BOUNDARY_ESCALATION_PAGE_CACHE_DIR = Path("data/cache/boundary_escalation_pages")


class BoundaryEscalationOutput(BaseModel):
    """Structured output expected from the vision-review model.

    ``llm_``-prefixed fields, same convention as ``scripts.
    validate_parsing_quality_sample.LLMValidationOutput``, to keep this
    model's own output distinct from the domain's [BoundaryReview] it gets
    mapped into.
    """

    llm_boundary_confirmed: bool = Field(
        ...,
        description=(
            "Whether the deterministic parser's claimed page range for this "
            "clause is correct in the attached pages -- not merged with a "
            "neighboring clause, not cut off mid-content."
        ),
    )
    llm_corrected_page_start: int | None = Field(
        None,
        description=(
            "If not confirmed, the correct first page for this clause. "
            "Must be one of the attached page numbers."
        ),
    )
    llm_corrected_page_end: int | None = Field(
        None,
        description=(
            "If not confirmed, the correct last page for this clause. "
            "Must be one of the attached page numbers."
        ),
    )
    llm_split_suggested: bool = Field(
        False,
        description=(
            "Whether the attached pages show this clause actually containing "
            "two or more distinct numbered sub-clauses that should be split "
            "apart, independent of whether the outer boundary is correct."
        ),
    )
    llm_split_notes: str = Field(
        "",
        description="If a split is suggested, a short note on where and why.",
    )
    llm_reasoning: str = Field(..., description="Short overall rationale.")


class VisionCallStats:
    """Mutable call/token/time accumulator for one reviewer instance."""

    def __init__(self) -> None:
        """Start every counter at zero."""
        self.call_count = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_seconds = 0.0


def _build_boundary_review_message(
    clause_title: str,
    claimed_page_start: int,
    claimed_page_end: int,
    page_images: tuple[bytes, ...],
) -> list[str | dict[str, object]]:
    """Build the multimodal human-turn content: the parser's claim plus pages."""
    prompt_text = (
        "You are reviewing the boundary an automated Brazilian insurance "
        "policy clause parser assigned to one clause, against the attached "
        "page images. The pages are a window around the parser's claim: "
        "one page of margin before the claimed start and after the claimed "
        "end, so you can judge whether the boundary is correct, truncated, "
        "or merged with a neighboring clause.\n\n"
        f"Parser's claim:\n"
        f"- Clause title: {clause_title}\n"
        f"- Claimed page range: {claimed_page_start}-{claimed_page_end}\n\n"
        "Judge, independently of the parser's claim:\n"
        "1. Does the clause actually start and end on the claimed pages, or "
        "is it truncated, merged with a neighbor, or misattributed?\n"
        "2. If not, what is the correct page range? It must be one of the "
        "attached pages.\n"
        "3. Does this clause actually contain two or more distinct numbered "
        "sub-clauses that should be split apart? This is independent of "
        "whether the outer boundary above is correct.\n"
        "Provide a short rationale tying these judgments together."
    )
    content: list[str | dict[str, object]] = [{"type": "text", "text": prompt_text}]
    for image_bytes in page_images:
        encoded = base64.b64encode(image_bytes).decode("ascii")
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{encoded}"},
            }
        )
    return content


class LangchainBoundaryVisionReviewer(BoundaryVisionReviewerPort):
    """Uses a Langchain vision-capable BaseChatModel to review boundaries."""

    def __init__(self, llm: BaseChatModel) -> None:
        """Wrap ``llm``, tracking call/token/time stats as reviews run."""
        self._chain = llm.with_structured_output(
            BoundaryEscalationOutput, include_raw=True
        )
        self.stats = VisionCallStats()

    def review(
        self,
        *,
        clause_title: str,
        claimed_page_start: int,
        claimed_page_end: int,
        page_images: tuple[bytes, ...],
    ) -> BoundaryReview:
        """Send the claim and page images to the model, returning its judgment."""
        content = _build_boundary_review_message(
            clause_title, claimed_page_start, claimed_page_end, page_images
        )
        start = time.monotonic()
        raw = cast(
            dict[str, object], self._chain.invoke([HumanMessage(content=content)])
        )
        self.stats.total_seconds += time.monotonic() - start
        self.stats.call_count += 1

        usage = getattr(raw.get("raw"), "usage_metadata", None) or {}
        self.stats.total_input_tokens += int(usage.get("input_tokens", 0) or 0)
        self.stats.total_output_tokens += int(usage.get("output_tokens", 0) or 0)

        parsed = cast(BoundaryEscalationOutput, raw["parsed"])
        return BoundaryReview(
            confirmed=parsed.llm_boundary_confirmed,
            corrected_page_start=parsed.llm_corrected_page_start,
            corrected_page_end=parsed.llm_corrected_page_end,
            split_suggested=parsed.llm_split_suggested,
            split_notes=parsed.llm_split_notes,
            reasoning=parsed.llm_reasoning,
        )


class PyMuPdfPageRasterizer(PageRasterizerPort):
    """Rasterizes PDF pages to PNG bytes, cached on disk by page number."""

    def __init__(self, dpi: int) -> None:
        """Configure rasterization DPI -- same setting OCR/M1-08b use."""
        self._dpi = dpi

    def rasterize(
        self, pdf_path: Path, document_id: str, first_page: int, last_page: int
    ) -> tuple[bytes, ...]:
        """Rasterize ``pdf_path``'s pages ``[first_page, last_page]``, cached."""
        cache_dir = BOUNDARY_ESCALATION_PAGE_CACHE_DIR / document_id
        targets = {
            page_num: cache_dir / f"page_{page_num:03d}.png"
            for page_num in range(first_page, last_page + 1)
        }
        if not all(path.exists() for path in targets.values()):
            cache_dir.mkdir(parents=True, exist_ok=True)
            document = fitz.open(pdf_path)
            try:
                zoom = self._dpi / 72
                matrix = fitz.Matrix(zoom, zoom)
                for page_num, png_path in targets.items():
                    if png_path.exists():
                        continue
                    page = document.load_page(page_num - 1)
                    pixmap = page.get_pixmap(matrix=matrix)
                    pixmap.save(png_path)
            finally:
                document.close()
        return tuple(targets[page_num].read_bytes() for page_num in sorted(targets))
