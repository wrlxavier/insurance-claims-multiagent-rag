from pathlib import Path
from typing import Any, cast

import fitz
import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable, RunnableLambda

from domain.boundary_escalation import BoundaryReview
from infrastructure.parsing import boundary_vision
from infrastructure.parsing.boundary_vision import (
    BoundaryEscalationOutput,
    LangchainBoundaryVisionReviewer,
    PyMuPdfPageRasterizer,
)


class _FakeRawMessage:
    """Stand-in for the AIMessage returned alongside a structured parse."""

    def __init__(self, usage_metadata: dict[str, int] | None) -> None:
        self.usage_metadata = usage_metadata


class FakeChatModel:
    """Stand-in for a Langchain ``BaseChatModel`` supporting ``include_raw=True``."""

    def __init__(
        self,
        parsed: BoundaryEscalationOutput,
        usage_metadata: dict[str, int] | None = None,
    ) -> None:
        self.parsed = parsed
        self.usage_metadata = usage_metadata
        self.received_content: list[object] = []

    def with_structured_output(
        self, schema: type, include_raw: bool = False
    ) -> Runnable[Any, Any]:
        def _invoke(messages: Any) -> dict[str, object]:
            self.received_content.append(messages)
            return {"parsed": self.parsed, "raw": _FakeRawMessage(self.usage_metadata)}

        return RunnableLambda(_invoke)


@pytest.mark.unit
def test_review_maps_parsed_fields_into_boundary_review() -> None:
    parsed = BoundaryEscalationOutput(
        llm_boundary_confirmed=False,
        llm_corrected_page_start=4,
        llm_corrected_page_end=6,
        llm_split_suggested=True,
        llm_split_notes="split here",
        llm_reasoning="because",
    )
    fake_llm = cast(
        BaseChatModel, FakeChatModel(parsed, {"input_tokens": 100, "output_tokens": 20})
    )
    reviewer = LangchainBoundaryVisionReviewer(fake_llm)

    result = reviewer.review(
        clause_title="Riscos Excluídos",
        claimed_page_start=3,
        claimed_page_end=5,
        page_images=(b"fake-png-bytes",),
    )

    assert result == BoundaryReview(
        confirmed=False,
        corrected_page_start=4,
        corrected_page_end=6,
        split_suggested=True,
        split_notes="split here",
        reasoning="because",
    )
    assert reviewer.stats.call_count == 1
    assert reviewer.stats.total_input_tokens == 100
    assert reviewer.stats.total_output_tokens == 20
    assert reviewer.stats.total_seconds >= 0.0


@pytest.mark.unit
def test_review_accumulates_stats_across_calls() -> None:
    parsed = BoundaryEscalationOutput(llm_boundary_confirmed=True, llm_reasoning="ok")
    fake_llm = cast(
        BaseChatModel, FakeChatModel(parsed, {"input_tokens": 10, "output_tokens": 5})
    )
    reviewer = LangchainBoundaryVisionReviewer(fake_llm)

    reviewer.review(
        clause_title="A", claimed_page_start=1, claimed_page_end=1, page_images=(b"x",)
    )
    reviewer.review(
        clause_title="B", claimed_page_start=2, claimed_page_end=2, page_images=(b"y",)
    )

    assert reviewer.stats.call_count == 2
    assert reviewer.stats.total_input_tokens == 20
    assert reviewer.stats.total_output_tokens == 10


@pytest.mark.unit
def test_review_tolerates_missing_usage_metadata() -> None:
    parsed = BoundaryEscalationOutput(llm_boundary_confirmed=True, llm_reasoning="ok")
    fake_llm = cast(BaseChatModel, FakeChatModel(parsed, usage_metadata=None))
    reviewer = LangchainBoundaryVisionReviewer(fake_llm)

    reviewer.review(
        clause_title="A", claimed_page_start=1, claimed_page_end=1, page_images=(b"x",)
    )

    assert reviewer.stats.call_count == 1
    assert reviewer.stats.total_input_tokens == 0
    assert reviewer.stats.total_output_tokens == 0


def _make_pdf(tmp_path: Path, page_count: int) -> Path:
    """Write a minimal real PDF (blank pages), same pattern as test_ocr.py."""
    document = fitz.open()
    for _ in range(page_count):
        document.new_page(width=100.0, height=50.0)
    path = tmp_path / "synthetic.pdf"
    document.save(path)
    document.close()
    return path


@pytest.mark.unit
def test_rasterize_returns_one_png_per_requested_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        boundary_vision, "BOUNDARY_ESCALATION_PAGE_CACHE_DIR", tmp_path / "cache"
    )
    pdf_path = _make_pdf(tmp_path, page_count=4)

    rasterizer = PyMuPdfPageRasterizer(dpi=72)
    images = rasterizer.rasterize(pdf_path, "doc-1", 2, 3)

    assert len(images) == 2
    for image_bytes in images:
        assert image_bytes[:8] == b"\x89PNG\r\n\x1a\n"


@pytest.mark.unit
def test_rasterize_caches_pages_on_disk_and_survives_source_deletion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(
        boundary_vision, "BOUNDARY_ESCALATION_PAGE_CACHE_DIR", cache_dir
    )
    pdf_path = _make_pdf(tmp_path, page_count=1)

    rasterizer = PyMuPdfPageRasterizer(dpi=72)
    rasterizer.rasterize(pdf_path, "doc-1", 1, 1)

    cached_png = cache_dir / "doc-1" / "page_001.png"
    assert cached_png.exists()

    pdf_path.unlink()
    images = rasterizer.rasterize(pdf_path, "doc-1", 1, 1)

    assert len(images) == 1
    assert images[0] == cached_png.read_bytes()
