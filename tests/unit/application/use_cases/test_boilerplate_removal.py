import pytest

from application.use_cases.boilerplate_removal import (
    find_header_footer_keys,
    find_marketing_pages,
    find_toc_pages,
    front_matter_window,
    is_dot_leader_toc_line,
    remove_boilerplate,
    remove_headers_and_footers,
)
from domain.extracted_text import ExtractedDocument, ExtractedPage, ExtractedSpan


def _span(
    *,
    page_number: int,
    line_id: int,
    order: int,
    text: str,
    bbox: tuple[float, float, float, float] = (50.0, 100.0, 400.0, 110.0),
    font_size: float = 9.0,
) -> ExtractedSpan:
    return ExtractedSpan(
        document_id="d1",
        page_number=page_number,
        line_id=line_id,
        order=order,
        bbox=bbox,
        font_size=font_size,
        font_name="Test",
        text=text,
    )


def _page(
    page_number: int, spans: list[ExtractedSpan], *, char_count: int | None = None
) -> ExtractedPage:
    spans_t = tuple(spans)
    if char_count is None:
        char_count = sum(len(span.text) for span in spans_t)
    return ExtractedPage(page_number=page_number, spans=spans_t, char_count=char_count)


def _empty_page(page_number: int) -> ExtractedPage:
    return ExtractedPage(page_number=page_number, spans=(), char_count=0)


def _document(pages: list[ExtractedPage]) -> ExtractedDocument:
    return ExtractedDocument(
        document_id="d1", filename="f.pdf", pages=tuple(pages), extractor_version="v1"
    )


def _body_page(
    page_number: int, char_count: int = 3000, font_size: float = 12.0
) -> ExtractedPage:
    """A body page with a unique-per-page text, so it never coincides with
    another page's line and gets mistaken for a repeated header/footer."""
    text = f"pagina {page_number} " + "x" * char_count
    span = _span(
        page_number=page_number, line_id=0, order=0, text=text, font_size=font_size
    )
    return _page(page_number, [span], char_count=len(text))


def _toc_like_page(page_number: int, num_entries: int = 10) -> ExtractedPage:
    """A page whose lines all end in a short number, dot-leader style."""
    spans = [
        _span(
            page_number=page_number,
            line_id=i,
            order=i,
            text=f"Section {i} .......... {i + 1}",
            bbox=(50.0, 100.0 + i * 15, 300.0, 110.0 + i * 15),
        )
        for i in range(num_entries)
    ]
    return _page(page_number, spans)


# ---------------------------------------------------------------------------
# is_dot_leader_toc_line
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_is_dot_leader_toc_line_true_for_classic_dot_leader() -> None:
    assert is_dot_leader_toc_line(
        "GLOSSÁRIO ............................................................. 3"
    )


@pytest.mark.unit
def test_is_dot_leader_toc_line_false_for_space_padded_toc_entry() -> None:
    # Real Mapfre TOC entries use no dots at all -- caught by find_toc_pages
    # instead, not by this per-line regex.
    text = (
        "3.        ANÁLISE DA PROPOSTA, CONTRATAÇÃO DO SEGURO E TRANSFERÊNCIA DO "
        "SEGURO 14"
    )
    assert not is_dot_leader_toc_line(text)


@pytest.mark.unit
def test_is_dot_leader_toc_line_false_for_ordinary_prose() -> None:
    assert not is_dot_leader_toc_line(
        "Fica garantido ao Segurado o pagamento dos prejuízos sofridos."
    )


# ---------------------------------------------------------------------------
# front_matter_window
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    ("page_count", "expected_window"),
    [
        (207, 10),  # Bradesco
        (146, 7),  # Mapfre
        (122, 6),  # Youse
        (9, 3),  # shortest fixture in the corpus -- floor applies
        (2, 2),  # window must never exceed the document itself
    ],
)
def test_front_matter_window(page_count: int, expected_window: int) -> None:
    assert front_matter_window(page_count) == expected_window


# ---------------------------------------------------------------------------
# headers/footers
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_find_header_footer_keys_separates_true_header_from_coincidence() -> None:
    """Models the real Youse corpus finding: a body clause number ("2.2")
    recurred at the same y-position on 64% of pages by pure coincidence,
    while a true header hit 95%+. The 0.9 threshold must keep the header
    and drop the coincidence."""
    pages = []
    for page_number in range(1, 101):
        spans = []
        if page_number <= 95:
            spans.append(
                _span(
                    page_number=page_number,
                    line_id=0,
                    order=0,
                    text="Bradesco Seguro Auto",
                    bbox=(50.0, 36.0, 200.0, 46.0),
                )
            )
        if page_number <= 64:
            spans.append(
                _span(
                    page_number=page_number,
                    line_id=1,
                    order=1,
                    text="2.2",
                    bbox=(50.0, 54.0, 70.0, 64.0),
                )
            )
        pages.append(_page(page_number, spans))
    document = _document(pages)

    keys = find_header_footer_keys(document)

    assert any(key[0] == "Bradesco Seguro Auto" for key in keys)
    assert not any(key[0] == "2.2" for key in keys)


@pytest.mark.unit
def test_remove_headers_and_footers_drops_only_the_true_header() -> None:
    pages = []
    for page_number in range(1, 101):
        spans = [
            _span(
                page_number=page_number,
                line_id=0,
                order=0,
                text="Bradesco Seguro Auto",
                bbox=(50.0, 36.0, 200.0, 46.0),
            )
        ]
        if page_number <= 64:
            spans.append(
                _span(
                    page_number=page_number,
                    line_id=1,
                    order=1,
                    text="2.2",
                    bbox=(50.0, 54.0, 70.0, 64.0),
                )
            )
        pages.append(_page(page_number, spans))
    document = _document(pages)

    cleaned, lines_removed = remove_headers_and_footers(document)

    remaining_headers = [
        span
        for page in cleaned.pages
        for span in page.spans
        if span.text == "Bradesco Seguro Auto"
    ]
    remaining_coincidences = [
        span for page in cleaned.pages for span in page.spans if span.text == "2.2"
    ]
    assert remaining_headers == []
    assert len(remaining_coincidences) == 64
    assert lines_removed == 100


# ---------------------------------------------------------------------------
# TOC pages
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_find_toc_pages_detects_toc_and_excludes_transition_page() -> None:
    """Page 3 models the real Youse page-4 transition: a ratio (~0.29,
    2 of 7 lines) close to but below threshold, adjacent to real TOC
    pages -- it must not be swept in just because it's next door."""
    transition_texts = [
        "CLÁUSULA 1a - INFORMAÇÕES PRELIMINARES",
        "1.1 Este plano de seguro é garantido pela Seguradora CNPJ 12345678000199",
        "1.2 A aceitação da proposta está sujeita à análise do risco",
        "1.3 O registro do produto é automático",
        "1.4 As condições contratuais estão disponíveis no site",
        "1.5 O segurado poderá consultar a situação em 2024",
        "1.6 Para casos não previstos aplica-se a lei vigente",
    ]
    transition_spans = [
        _span(
            page_number=3,
            line_id=i,
            order=i,
            text=text,
            bbox=(50.0, 100.0 + i * 15, 400.0, 110.0 + i * 15),
        )
        for i, text in enumerate(transition_texts)
    ]
    pages = [
        _toc_like_page(1),
        _toc_like_page(2),
        _page(3, transition_spans),
    ]
    pages += [_empty_page(n) for n in range(4, 61)]
    document = _document(pages)

    assert find_toc_pages(document) == frozenset({1, 2})


# ---------------------------------------------------------------------------
# marketing/cover pages
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_find_marketing_pages_detects_sparse_large_font_cover_page() -> None:
    cover_span = _span(
        page_number=1,
        line_id=0,
        order=0,
        text="x" * 150,
        bbox=(50.0, 300.0, 500.0, 340.0),
        font_size=32.0,
    )
    pages = [_page(1, [cover_span], char_count=150)]
    pages += [_body_page(n) for n in range(2, 61)]
    document = _document(pages)

    assert find_marketing_pages(document) == frozenset({1})


@pytest.mark.unit
def test_find_marketing_pages_requires_both_sparse_and_large_font() -> None:
    """Models the real Bradesco negative case: page 1's max font is 1.7x
    the modal font (would trigger a font-ratio-only rule) but the page is
    dense, not sparse -- it must not be flagged as marketing."""
    dense_span = _span(
        page_number=1,
        line_id=0,
        order=0,
        text="x" * 8000,
        font_size=15.5,
    )
    pages = [_page(1, [dense_span], char_count=8000)]
    pages += [_body_page(n, char_count=4000, font_size=9.0) for n in range(2, 61)]
    document = _document(pages)

    assert find_marketing_pages(document) == frozenset()


@pytest.mark.unit
def test_find_marketing_pages_skips_pages_with_no_spans() -> None:
    """A page already emptied by an earlier stage must not crash detection
    (no spans means no max font) and must not be flagged again."""
    pages = [_empty_page(1)]
    pages += [_body_page(n) for n in range(2, 61)]
    document = _document(pages)

    assert find_marketing_pages(document) == frozenset()


# ---------------------------------------------------------------------------
# ordering and end-to-end
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_remove_boilerplate_strips_footer_before_computing_toc_ratio() -> None:
    """A page whose TOC entry-ratio only clears 0.4 if a repeated bare
    footer page-number line is still counted (1/3 without it, 2/4 with
    it). remove_boilerplate must strip headers/footers first, so this
    page is correctly left alone."""
    borderline_texts = [
        "Body sentence one without a number",
        "Body sentence two mentions ano 2024",
        "Body sentence three no digits",
    ]
    pages = []
    for page_number in range(1, 21):
        spans = [
            _span(
                page_number=page_number,
                line_id=i,
                order=i,
                text=text,
                bbox=(50.0, 100.0 + i * 15, 400.0, 110.0 + i * 15),
            )
            for i, text in enumerate(borderline_texts)
        ]
        spans.append(
            _span(
                page_number=page_number,
                line_id=99,
                order=99,
                text=str(page_number),
                bbox=(50.0, 800.0, 60.0, 810.0),
            )
        )
        pages.append(_page(page_number, spans))
    document = _document(pages)

    _cleaned, counts = remove_boilerplate(document)

    assert counts.header_footer_lines_removed > 0
    assert counts.toc_pages_removed == 0
    assert 1 not in counts.removed_page_numbers


@pytest.mark.unit
def test_remove_boilerplate_never_drops_glossary_content() -> None:
    glossary_spans = [
        _span(
            page_number=10,
            line_id=0,
            order=0,
            text="GLOSSÁRIO",
            bbox=(50.0, 100.0, 200.0, 120.0),
            font_size=14.0,
        ),
        _span(
            page_number=10,
            line_id=1,
            order=1,
            text="Aceitação: É a aprovação da proposta apresentada pelo Segurado.",
            bbox=(50.0, 130.0, 400.0, 150.0),
        ),
        _span(
            page_number=10,
            line_id=2,
            order=2,
            text="Acidente: Acontecimento súbito e imprevisto do qual resultem danos.",
            bbox=(50.0, 160.0, 400.0, 180.0),
        ),
    ]
    cover_span = _span(
        page_number=1,
        line_id=0,
        order=0,
        text="x" * 150,
        bbox=(50.0, 300.0, 500.0, 340.0),
        font_size=32.0,
    )
    pages = [
        _page(1, [cover_span], char_count=150),
        _toc_like_page(2),
        _toc_like_page(3),
    ]
    for page_number in range(4, 31):
        if page_number == 10:
            pages.append(_page(10, glossary_spans))
        else:
            pages.append(_body_page(page_number))
    document = _document(pages)

    cleaned, counts = remove_boilerplate(document)

    original_glossary = next(p for p in document.pages if p.page_number == 10)
    cleaned_glossary = next(p for p in cleaned.pages if p.page_number == 10)
    assert cleaned_glossary.spans == original_glossary.spans
    assert cleaned_glossary.char_count == original_glossary.char_count
    assert 1 in counts.removed_page_numbers
    assert {2, 3}.issubset(counts.removed_page_numbers)
