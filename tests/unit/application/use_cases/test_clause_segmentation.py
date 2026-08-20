from typing import Any

import pytest

from application.use_cases.clause_segmentation import (
    find_oversized_clauses,
    segment_document,
)
from domain.clause_tree import (
    Clause,
    ClauseTree,
    HeadingConvention,
    OrphanTextExceedsThresholdError,
)
from domain.extracted_text import ExtractedDocument, ExtractedPage, ExtractedSpan


def _span(
    *,
    page_number: int,
    line_id: int,
    order: int,
    text: str,
    x0: float = 50.0,
    y0: float = 100.0,
    font_size: float = 11.0,
    bold: bool = False,
) -> ExtractedSpan:
    return ExtractedSpan(
        document_id="d1",
        page_number=page_number,
        line_id=line_id,
        order=order,
        bbox=(x0, y0, x0 + max(len(text), 1) * 6.0, y0 + 10.0),
        font_size=font_size,
        font_name="Test-BoldMT" if bold else "TestMT",
        text=text,
    )


def _heading_line(
    page_number: int, line_id: int, text: str, *, x0: float = 50.0
) -> ExtractedSpan:
    """A fully-bold logical line -- the shape every real heading takes in the corpus."""
    return _span(
        page_number=page_number,
        line_id=line_id,
        order=line_id,
        text=text,
        x0=x0,
        bold=True,
    )


def _body_line(
    page_number: int, line_id: int, text: str, *, x0: float = 50.0
) -> ExtractedSpan:
    """A fully non-bold logical line -- ordinary body/content text."""
    return _span(
        page_number=page_number,
        line_id=line_id,
        order=line_id,
        text=text,
        x0=x0,
        bold=False,
    )


def _mixed_prefix_line(
    page_number: int, line_id: int, prefix: str, rest: str, *, x0: float = 50.0
) -> list[ExtractedSpan]:
    """A line whose only bold run is a short numeric prefix -- a numbered body
    paragraph's opening line, e.g. '1.1 A Aceitação do seguro está sujeita...'.
    Two spans on the same line_id, so bold_fraction dilutes below the gate."""
    return [
        _span(
            page_number=page_number,
            line_id=line_id,
            order=0,
            text=prefix,
            x0=x0,
            bold=True,
        ),
        _span(
            page_number=page_number,
            line_id=line_id,
            order=1,
            text=rest,
            x0=x0 + len(prefix) * 6.0,
            bold=False,
        ),
    ]


def _with_y0(span: ExtractedSpan, *, y0: float) -> ExtractedSpan:
    x0, old_y0, x1, y1 = span.bbox
    height = y1 - old_y0
    return ExtractedSpan(
        document_id=span.document_id,
        page_number=span.page_number,
        line_id=span.line_id,
        order=span.order,
        bbox=(x0, y0, x1, y0 + height),
        font_size=span.font_size,
        font_name=span.font_name,
        text=span.text,
    )


def _page(page_number: int, spans: list[ExtractedSpan]) -> ExtractedPage:
    spans_t = tuple(spans)
    return ExtractedPage(
        page_number=page_number,
        spans=spans_t,
        char_count=sum(len(s.text) for s in spans_t),
    )


def _document(pages: list[ExtractedPage]) -> ExtractedDocument:
    return ExtractedDocument(
        document_id="d1", filename="f.pdf", pages=tuple(pages), extractor_version="v1"
    )


def _ocr_page(page_number: int, text: str) -> ExtractedPage:
    span = ExtractedSpan(
        document_id="d1",
        page_number=page_number,
        line_id=0,
        order=0,
        bbox=(0.0, 0.0, 600.0, 800.0),
        font_size=0.0,
        font_name="",
        text=text,
    )
    return ExtractedPage(page_number=page_number, spans=(span,), char_count=len(text))


def _find(tree: ClauseTree, numbering_label: str) -> Clause:
    return next(c for c in tree.all_clauses if c.numbering_label == numbering_label)


@pytest.mark.unit
def test_subset_font_with_no_bold_in_name_falls_back_to_dominant_font() -> None:
    """doc 4 (Sura) real case: subset CID fonts carry no "bold" in their
    name at all -- headings use a dedicated font ("CIDFont+F2") exclusive
    of the body font ("CIDFont+F1"), so the document-level fallback must
    treat "not the dominant font" as the heavy-weight signal instead."""

    def _cid_span(
        page_number: int, line_id: int, text: str, *, font: str
    ) -> ExtractedSpan:
        return ExtractedSpan(
            document_id="d1",
            page_number=page_number,
            line_id=line_id,
            order=line_id,
            bbox=(
                50.0,
                100.0 + line_id * 15.0,
                50.0 + len(text) * 6.0,
                110.0 + line_id * 15.0,
            ),
            font_size=12.0,
            font_name=font,
            text=text,
        )

    spans = [
        _cid_span(1, 0, "1. Disposições Preliminares", font="CIDFont+F2"),
        _cid_span(
            1, 1, "A aceitação do Seguro estará sujeita à análise.", font="CIDFont+F1"
        ),
        _cid_span(1, 2, "2. Objetivo do Seguro", font="CIDFont+F2"),
        _cid_span(
            1, 3, "As Coberturas constantes garantem o pagamento.", font="CIDFont+F1"
        ),
    ]
    # Body font dominates by character count, as in the real document.
    spans.append(
        _cid_span(1, 4, "Corpo adicional em CIDFont+F1 " * 5, font="CIDFont+F1")
    )
    document = _document([_page(1, spans)])

    tree = segment_document(document)

    assert tree.report.clause_count == 2
    assert [c.numbering_label for c in tree.roots] == ["1", "2"]
    assert tree.report.orphan_ratio == 0.0


@pytest.mark.unit
def test_single_font_document_falls_back_to_font_size_delta() -> None:
    """[M1-04b] doc 5 (KOVR) real case: no "bold" font, and unlike doc 4 no
    dedicated second font either -- headings and body share one font name
    throughout (99.4% of the real document's characters), differing only
    by size (12.0pt headings vs 10.66pt body, a 12.6% delta). Before this
    fallback tier, the real document recovered zero clauses at all."""

    def _size_span(
        page_number: int, line_id: int, text: str, *, font_size: float, y0: float
    ) -> ExtractedSpan:
        return ExtractedSpan(
            document_id="d1",
            page_number=page_number,
            line_id=line_id,
            order=line_id,
            bbox=(50.0, y0, 50.0 + len(text) * 6.0, y0 + 10.0),
            font_size=font_size,
            font_name="NewJuneRegular",
            text=text,
        )

    # Realistic paragraph spacing: 15pt between ordinary body lines, a wider
    # 30pt gap before each section heading -- matching real PDF layout, and
    # clearing [MIN_HEADING_GAP_RATIO] against the 15pt baseline (unlike a
    # wrap-continuation line, which stays at the tight baseline gap; see
    # [_has_heading_position_signal]).
    spans = [
        _size_span(1, 0, "1) DISPOSIÇÕES INICIAIS", font_size=12.0, y0=100.0),
        _size_span(
            1,
            1,
            "A aceitação da proposta de seguro está sujeita à análise.",
            font_size=10.66,
            y0=115.0,
        ),
        _size_span(1, 2, "2) OBJETIVO DO SEGURO", font_size=12.0, y0=145.0),
        _size_span(
            1,
            3,
            "O objetivo do seguro é garantir o pagamento de indenização.",
            font_size=10.66,
            y0=160.0,
        ),
    ]
    # Body size dominates by character count, as in the real document.
    spans.append(
        _size_span(
            1, 4, "Corpo adicional em corpo normal " * 5, font_size=10.66, y0=175.0
        )
    )
    document = _document([_page(1, spans)])

    tree = segment_document(document)

    assert tree.report.clause_count == 2
    assert [c.title for c in tree.roots] == [
        "1) DISPOSIÇÕES INICIAIS",
        "2) OBJETIVO DO SEGURO",
    ]
    assert tree.report.orphan_ratio == 0.0


# ---------------------------------------------------------------------------
# 1. Plain decimal depth changes
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_plain_decimal_depth_changes_build_correct_parentage() -> None:
    document = _document(
        [
            _page(
                1,
                [
                    _heading_line(1, 0, "1. OBJETIVO DO SEGURO"),
                    _body_line(1, 1, "Corpo do item 1."),
                    _heading_line(1, 2, "1.1 Coberturas Básicas"),
                    _heading_line(1, 3, "1.1.1 Colisão"),
                    _heading_line(1, 4, "1.2 Coberturas Adicionais"),
                    _heading_line(1, 5, "2. ÂMBITO GEOGRÁFICO"),
                ],
            )
        ]
    )

    tree = segment_document(document)

    assert [c.numbering_label for c in tree.roots] == ["1", "2"]
    node_1 = _find(tree, "1")
    node_11 = _find(tree, "1.1")
    node_111 = _find(tree, "1.1.1")
    node_12 = _find(tree, "1.2")
    assert node_11.parent_id == node_1.clause_id
    assert node_111.parent_id == node_11.clause_id
    # 1.2 must be a sibling of 1.1, not nested under 1.1.1.
    assert node_12.parent_id == node_1.clause_id
    assert node_1.depth == 1
    assert node_11.depth == 2
    assert node_111.depth == 3
    assert node_1.content_lines == ("Corpo do item 1.",)
    assert tree.report.max_depth == 3


@pytest.mark.unit
def test_numbered_heading_title_wrap_does_not_open_a_spurious_part() -> None:
    """doc 28's real case: '7. RISCOS EXCLUÍDOS/ BENS E INTERESSES' wraps
    onto 'NÃO GARANTIDOS' on the next line -- that continuation must join
    the same title, not spuriously reset the stack as a new part header."""
    document = _document(
        [
            _page(
                1,
                [
                    _heading_line(1, 0, "7. RISCOS EXCLUÍDOS/ BENS E INTERESSES"),
                    _heading_line(1, 1, "NÃO GARANTIDOS"),
                    _heading_line(1, 2, "8. CUSTOS"),
                ],
            )
        ]
    )

    tree = segment_document(document)

    node_7 = _find(tree, "7")
    assert node_7.title == "7. RISCOS EXCLUÍDOS/ BENS E INTERESSES NÃO GARANTIDOS"
    assert len(tree.roots) == 2
    assert [c.numbering_label for c in tree.roots] == ["7", "8"]


# ---------------------------------------------------------------------------
# 1b. Bare numeral and its title split across two separate logical lines
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_bare_numeral_and_title_on_separate_lines_join_into_one_heading() -> None:
    """doc 15 (Mapfre) real case: a top-level clause's numeral and its bold
    title render as two separate logical lines -- a bare "1." alone, then
    "COBERTURA BÁSICA COMPREENSIVA..." on the next line. Neither
    [_NUMBERED_DECIMAL_TOP] nor [_NUMBERED_DECIMAL_DEEP] can match a bare
    numeral (both require title text on the same matched line), so without
    [_try_join_bare_numeral_title] the numeral falls through as content and
    the title independently opens as a spurious UNNUMBERED_PART root
    instead of becoming that numeral's own numbered heading."""
    document = _document(
        [
            _page(
                1,
                [
                    _heading_line(1, 0, "1."),
                    _heading_line(
                        1, 1, "COBERTURA BÁSICA COMPREENSIVA COLISÃO, INCÊNDIO"
                    ),
                    _heading_line(1, 2, "2. COBERTURAS ADICIONAIS"),
                ],
            )
        ]
    )

    tree = segment_document(document)

    assert len(tree.roots) == 2
    node_1 = _find(tree, "1")
    assert node_1.convention == HeadingConvention.NUMBERED_DECIMAL
    assert node_1.title == "1. COBERTURA BÁSICA COMPREENSIVA COLISÃO, INCÊNDIO"
    assert [c.numbering_label for c in tree.roots] == ["1", "2"]


@pytest.mark.unit
def test_bare_numeral_without_a_following_title_line_stays_as_content() -> None:
    """Negative case for [_try_join_bare_numeral_title]: a bare numeral with
    nothing after it on the page must not crash and must not join anything
    -- it just falls through as ordinary content, same as before this fix,
    since neither numbered-decimal regex matches a numeral alone."""
    document = _document(
        [
            _page(
                1,
                [
                    _heading_line(1, 0, "1. OBJETO DO SEGURO"),
                    _heading_line(1, 1, "2."),
                ],
            )
        ]
    )

    tree = segment_document(document)

    assert len(tree.roots) == 1
    assert _find(tree, "1").content_lines == ("2.",)


# ---------------------------------------------------------------------------
# 2. CLÁUSULA N form with sub-items reusing the clause's own number
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_clausula_form_nests_reused_number_subitems() -> None:
    document = _document(
        [
            _page(
                1,
                [
                    _heading_line(1, 0, "CLÁUSULA 10a – PAGAMENTO DO PRÊMIO"),
                    _heading_line(1, 1, "10.10 No caso de substituição do veículo"),
                    _heading_line(1, 2, "10.11 Pagamento em atraso"),
                ],
            )
        ]
    )

    tree = segment_document(document)

    clausula = _find(tree, "10")
    assert clausula.convention == HeadingConvention.CLAUSULA_KEYWORD
    assert clausula.depth == 1
    sub_10_10 = _find(tree, "10.10")
    sub_10_11 = _find(tree, "10.11")
    assert sub_10_10.parent_id == clausula.clause_id
    assert sub_10_11.parent_id == clausula.clause_id
    assert sub_10_10.depth == 2


# ---------------------------------------------------------------------------
# 3. Unnumbered ALL-CAPS part header
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_unnumbered_part_header_becomes_parent_of_following_numbered_heading() -> None:
    document = _document(
        [
            _page(
                1,
                [
                    _heading_line(1, 0, "DISPOSIÇÕES PRELIMINARES"),
                    _heading_line(1, 1, "1. OBJETO DO SEGURO"),
                ],
            )
        ]
    )

    tree = segment_document(document)

    assert len(tree.roots) == 1
    part = tree.roots[0]
    assert part.convention == HeadingConvention.UNNUMBERED_PART
    assert part.depth == 0
    numbered = _find(tree, "1")
    assert numbered.parent_id == part.clause_id
    assert numbered.depth == 1


# ---------------------------------------------------------------------------
# 3b. Position-gap gate on UNNUMBERED_PART detection
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_all_caps_body_sentence_wrap_is_not_detected_as_new_heading() -> None:
    """doc 15 (Mapfre) p59, samples #21/#22: in a whole-document-bold legacy
    template, bold fraction is saturated on nearly every line, leaving
    "<=10 words, all-uppercase" as the only UNNUMBERED_PART gate -- a
    wrapped continuation line of an ALL-CAPS sentence independently clears
    it and gets misdetected as a new heading, truncating the clause that
    should have captured it. [_has_heading_position_signal] adds a
    line-to-line vertical-spacing gate: these continuation lines sit at the
    document's ordinary (tight) line pitch, while the genuine heading's own
    gap is far wider."""
    lines_spec = [
        (_heading_line(1, 0, "1. OBJETO DO SEGURO"), 100.0),
        (_body_line(1, 1, "Texto de corpo comum um."), 115.0),
        (_body_line(1, 2, "Texto de corpo comum dois."), 130.0),
        (_body_line(1, 3, "Texto de corpo comum tres."), 145.0),
        (_body_line(1, 4, "Texto de corpo comum quatro."), 160.0),
        # 30pt gap: genuine heading.
        (_heading_line(1, 5, "COBERTURAS BÁSICAS"), 190.0),
        # A real body line intervenes here (matching the real corpus shape --
        # the false-positive lines below are mid-clause, not adjacent to the
        # heading), so [consume_title_wrap]'s title-wrap absorption -- a
        # different, unrelated mechanism for a heading's OWN wrapped title --
        # never runs; this exercises the main loop's own position gate.
        (_body_line(1, 6, "A contratação de qualquer cobertura básica."), 205.0),
        (
            _heading_line(1, 7, "SEM PREJUÍZO DAS DEMAIS CLÁUSULAS DESTE"),
            220.0,
        ),  # 15pt: wrap continuation, must not open a new root.
        (
            _heading_line(1, 8, "CONTRATO CELEBRADO ENTRE AS PARTES."),
            235.0,
        ),  # 15pt: wrap continuation, must not open a new root.
    ]
    spans = [_with_y0(span, y0=y0) for span, y0 in lines_spec]
    document = _document([_page(1, spans)])

    tree = segment_document(document)

    titles = {c.title for c in tree.all_clauses}
    assert "SEM PREJUÍZO DAS DEMAIS CLÁUSULAS DESTE" not in titles
    assert "CONTRATO CELEBRADO ENTRE AS PARTES." not in titles
    cobertura = next(c for c in tree.all_clauses if c.title == "COBERTURAS BÁSICAS")
    assert "SEM PREJUÍZO DAS DEMAIS CLÁUSULAS DESTE" in cobertura.content_lines
    assert "CONTRATO CELEBRADO ENTRE AS PARTES." in cobertura.content_lines


@pytest.mark.unit
def test_lettered_list_item_wrap_continuation_is_not_detected_as_new_heading() -> None:
    """doc 15 p141 sample #24: a line break inside item "V)" of a lettered
    exclusion list was falsely detected as a new UNNUMBERED_PART heading,
    fragmenting the enclosing clause. The list item's own first line is
    already recognized as content via [is_list_item_line] before heading
    detection even runs; its wrapped continuation line has no such prefix
    and must instead be caught by the same position-gap gate as an
    ALL-CAPS body-sentence wrap."""
    lines_spec = [
        (_heading_line(1, 0, "34. RISCOS EXCLUÍDOS"), 100.0),
        (_body_line(1, 1, "Texto de corpo comum um."), 115.0),
        (_body_line(1, 2, "Texto de corpo comum dois."), 130.0),
        (_body_line(1, 3, "Texto de corpo comum tres."), 145.0),
        (_heading_line(1, 4, "V) DOS RISCOS COBERTOS"), 175.0),  # 30pt gap.
        (_heading_line(1, 5, "OS EXCLUÍDOS PELA APÓLICE;"), 190.0),  # 15pt: wrap.
    ]
    spans = [_with_y0(span, y0=y0) for span, y0 in lines_spec]
    document = _document([_page(1, spans)])

    tree = segment_document(document)

    titles = {c.title for c in tree.all_clauses}
    assert "OS EXCLUÍDOS PELA APÓLICE;" not in titles
    parent = _find(tree, "34")
    assert "V) DOS RISCOS COBERTOS" in parent.content_lines
    assert "OS EXCLUÍDOS PELA APÓLICE;" in parent.content_lines


@pytest.mark.unit
def test_genuine_unnumbered_part_after_paragraph_break_is_still_detected() -> None:
    """Regression guard for [_has_heading_position_signal]: a genuine
    UNNUMBERED_PART title that follows a real paragraph break (a wide
    vertical gap, not a tight wrap) must still open its own root --
    protects real corpus cases like doc 10's "RESUMO DE COBERTURAS DA
    ASSISTÊNCIA À MOTOCICLETAS" and doc 28's wrapped part title from being
    swallowed by the new gate."""
    lines_spec = [
        (_body_line(1, 0, "Texto de corpo comum um."), 100.0),
        (_body_line(1, 1, "Texto de corpo comum dois."), 115.0),
        (_body_line(1, 2, "Texto de corpo comum tres."), 130.0),
        (_body_line(1, 3, "Texto de corpo comum quatro."), 145.0),
        (_heading_line(1, 4, "RESUMO DE COBERTURAS"), 180.0),  # 35pt gap.
        (_heading_line(1, 5, "1. OBJETO DO SEGURO"), 195.0),
    ]
    spans = [_with_y0(span, y0=y0) for span, y0 in lines_spec]
    document = _document([_page(1, spans)])

    tree = segment_document(document)

    assert len(tree.roots) == 1
    part = tree.roots[0]
    assert part.title == "RESUMO DE COBERTURAS"
    assert part.convention == HeadingConvention.UNNUMBERED_PART


@pytest.mark.unit
def test_page_top_unnumbered_part_heading_is_detected_without_a_preceding_gap() -> None:
    """Regression guard: the first line of a page has no previous same-page
    line to compare a gap against -- [_has_heading_position_signal] must
    still detect it (page-top headings are the corpus norm), not silently
    reject every part title that happens to start a page."""
    lines_spec = [
        (_body_line(1, 0, "Texto de corpo comum um."), 100.0),
        (_body_line(1, 1, "Texto de corpo comum dois."), 115.0),
        (_body_line(1, 2, "Texto de corpo comum tres."), 130.0),
        (_body_line(1, 3, "Texto de corpo comum quatro."), 145.0),
        (_body_line(1, 4, "Texto de corpo comum cinco."), 160.0),
        (_body_line(1, 5, "Texto de corpo comum seis."), 175.0),
    ]
    page1_spans = [_with_y0(span, y0=y0) for span, y0 in lines_spec]
    page2_spans = [_heading_line(2, 0, "DISPOSIÇÕES GERAIS")]
    document = _document([_page(1, page1_spans), _page(2, page2_spans)])

    tree = segment_document(document)

    assert len(tree.roots) == 1
    assert tree.roots[0].title == "DISPOSIÇÕES GERAIS"


@pytest.mark.unit
def test_narrow_gap_unnumbered_part_is_still_detected() -> None:
    """[M1-08b] doc 11 (AKAD) real case: the genuine home-insurance-rider
    heading "CONDIÇÕES ESPECIAIS - COBERTURAS PARA A RESIDÊNCIA - PROTEÇÃO
    COMBINADA" sits at a 20.76pt gap from the previous line vs. a 13.8pt
    baseline (ratio 1.504) -- short of the old MIN_HEADING_GAP_RATIO=1.6
    gate, silently merging the whole rider section into the preceding
    clause (`make parse`'s oversized-clause safeguard caught it as
    `11:membros-inferiores/7-7`, 19,032 chars). 1.45 admits this ratio
    while a wrap continuation at the ~13.8pt baseline (ratio ~1.0) still
    fails it."""
    lines_spec = [
        (_body_line(1, 0, "Texto de corpo comum um."), 100.0),
        (_body_line(1, 1, "Texto de corpo comum dois."), 113.8),
        (_body_line(1, 2, "Texto de corpo comum tres."), 127.6),
        (_body_line(1, 3, "Texto de corpo comum quatro."), 141.4),
        # 20.76pt gap (ratio 1.504 against the 13.8pt baseline above).
        (_heading_line(1, 4, "CONDIÇÕES ESPECIAIS COBERTURAS"), 162.16),
    ]
    spans = [_with_y0(span, y0=y0) for span, y0 in lines_spec]
    document = _document([_page(1, spans)])

    tree = segment_document(document)

    titles = {c.title for c in tree.all_clauses}
    assert "CONDIÇÕES ESPECIAIS COBERTURAS" in titles
    part = next(
        c for c in tree.all_clauses if c.title == "CONDIÇÕES ESPECIAIS COBERTURAS"
    )
    assert part.convention == HeadingConvention.UNNUMBERED_PART


@pytest.mark.unit
def test_ordinal_no_abbreviation_does_not_block_unnumbered_part_detection() -> None:
    """doc 13 (Zurich) sample #16: "COBERTURA No 41 - CARROCERIAS" etc. --
    PDF extraction renders the ordinal-indicator glyph "º" (as in "nº") as
    a bare lowercase "o", so the raw text reads "COBERTURA No 41", failing
    the all-uppercase UNNUMBERED_PART check on that single letter and
    silently merging 20 pages / 40,000+ characters into the preceding open
    clause. [_ORDINAL_NO_ABBREVIATION] normalizes just that token before
    the uppercase check."""
    lines_spec = [
        (_heading_line(1, 0, "RISCOS EXCLUÍDOS"), 100.0),
        (_body_line(1, 1, "Texto de corpo comum um."), 115.0),
        (_body_line(1, 2, "Texto de corpo comum dois."), 130.0),
        (_body_line(1, 3, "Texto de corpo comum tres."), 145.0),
        (_body_line(1, 4, "Texto de corpo comum quatro."), 160.0),
        (_heading_line(1, 5, "COBERTURA No 41 - CARROCERIAS"), 195.0),  # 35pt gap.
    ]
    spans = [_with_y0(span, y0=y0) for span, y0 in lines_spec]
    document = _document([_page(1, spans)])

    tree = segment_document(document)

    assert len(tree.roots) == 2
    assert tree.roots[1].title == "COBERTURA No 41 - CARROCERIAS"
    assert tree.roots[1].convention == HeadingConvention.UNNUMBERED_PART


# ---------------------------------------------------------------------------
# 4. Roman-numeral part header wrapped across multiple lines
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_roman_numeral_part_header_merges_wrapped_lines() -> None:
    document = _document(
        [
            _page(
                1,
                [
                    _heading_line(
                        1, 0, "I - CLÁUSULAS COMUNS ÀS COBERTURAS DOS SEGUROS DE"
                    ),
                    _heading_line(1, 1, "AUTOMÓVEL E ACIDENTES PESSOAIS"),
                    _heading_line(1, 2, "1. ACEITAÇÃO DO SEGURO"),
                ],
            )
        ]
    )

    tree = segment_document(document)

    assert len(tree.roots) == 1
    part = tree.roots[0]
    assert part.convention == HeadingConvention.UNNUMBERED_PART
    assert "AUTOMÓVEL E ACIDENTES PESSOAIS" in part.title
    assert part.title.startswith("I - CLÁUSULAS COMUNS")
    numbered = _find(tree, "1")
    assert numbered.parent_id == part.clause_id


# ---------------------------------------------------------------------------
# 5/6. Lettered and bulleted list attachment
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_lettered_list_items_attach_as_content_not_nodes() -> None:
    document = _document(
        [
            _page(
                1,
                [
                    _heading_line(1, 0, "TERCEIRO"),
                    _body_line(1, 1, "a) o próprio Segurado;"),
                    _body_line(1, 2, "b) o condutor e qualquer passageiro do veículo;"),
                    _body_line(1, 3, "c) o causador do Sinistro;"),
                ],
            )
        ]
    )

    tree = segment_document(document)

    assert tree.report.clause_count == 1
    node = tree.roots[0]
    assert node.content_lines == (
        "a) o próprio Segurado;",
        "b) o condutor e qualquer passageiro do veículo;",
        "c) o causador do Sinistro;",
    )


@pytest.mark.unit
def test_uppercase_lettered_list_items_attach_as_content_not_nodes() -> None:
    """doc 15's real exclusion list uses uppercase, double-letter-after-Z
    markers ('G)', 'AA)', 'GG)') -- each must attach as content, not
    fragment into its own spurious unnumbered-part node."""
    document = _document(
        [
            _page(
                1,
                [
                    _heading_line(1, 0, "26. RISCOS EXCLUÍDOS"),
                    _body_line(1, 1, "G) MORADIAS COLETIVAS E SIMILARES;"),
                    _body_line(1, 2, "V) ROUBO OU FURTO MEDIANTE ARROMBAMENTO;"),
                    _body_line(1, 3, "AA) CURTO-CIRCUITO NA REDE ELÉTRICA;"),
                    _body_line(1, 4, "GG) RESIDÊNCIAS SOB INTERDIÇÃO;"),
                ],
            )
        ]
    )

    tree = segment_document(document)

    assert tree.report.clause_count == 1
    node = tree.roots[0]
    assert node.content_lines == (
        "G) MORADIAS COLETIVAS E SIMILARES;",
        "V) ROUBO OU FURTO MEDIANTE ARROMBAMENTO;",
        "AA) CURTO-CIRCUITO NA REDE ELÉTRICA;",
        "GG) RESIDÊNCIAS SOB INTERDIÇÃO;",
    )


@pytest.mark.unit
def test_long_all_caps_sentence_without_lettered_marker_is_not_a_heading() -> None:
    """Brazilian insurance contracts legally must highlight limiting/
    exclusionary clauses in bold caps (CDC-style emphasis) -- a long capped
    sentence with no numbering must stay content, not become a part node."""
    document = _document(
        [
            _page(
                1,
                [
                    _heading_line(1, 0, "18. PERDA DE DIREITOS"),
                    _heading_line(
                        1,
                        1,
                        "RESPONSABILIZA PELOS VALORES QUE ULTRAPASSAR O LIMITE "
                        "SEGURADO CONTRATADO SENDO OS MESMOS DE RESPONSABILIDADE",
                    ),
                ],
            )
        ]
    )

    tree = segment_document(document)

    assert tree.report.clause_count == 1
    node = tree.roots[0]
    assert node.content_lines[0].startswith("RESPONSABILIZA PELOS VALORES")


@pytest.mark.unit
def test_bulleted_list_items_attach_as_content_not_nodes() -> None:
    document = _document(
        [
            _page(
                1,
                [
                    _heading_line(1, 0, "1. RISCOS EXCLUÍDOS"),
                    _body_line(1, 1, "• Armamentos"),
                    _body_line(1, 2, "• Cargas Inflamáveis"),
                    _body_line(1, 3, "- Combustíveis"),
                ],
            )
        ]
    )

    tree = segment_document(document)

    assert tree.report.clause_count == 1
    node = tree.roots[0]
    assert node.content_lines == (
        "• Armamentos",
        "• Cargas Inflamáveis",
        "- Combustíveis",
    )


# ---------------------------------------------------------------------------
# 7. Bold-fraction gate rejects a non-bold cross-reference matching the
#    numbering regex (doc 12's "CLAUSULA 2" false-positive risk).
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_non_bold_clausula_cross_reference_is_not_a_heading() -> None:
    document = _document(
        [
            _page(
                1,
                [
                    _heading_line(1, 0, "1. ASSISTÊNCIA 24 HORAS"),
                    _body_line(
                        1,
                        1,
                        "LIMITES: CONFORME DESCRITOS NA CLAUSULA 2 - PLANOS, PRODUTOS",
                    ),
                ],
            )
        ]
    )

    tree = segment_document(document)

    assert tree.report.clause_count == 1
    assert tree.report.orphan_char_count == 0
    node = tree.roots[0]
    assert node.content_lines == (
        "LIMITES: CONFORME DESCRITOS NA CLAUSULA 2 - PLANOS, PRODUTOS",
    )


# ---------------------------------------------------------------------------
# 8. Body-paragraph-opening line (numbered, but only its prefix is bold) is
#    not misdetected as a heading, even under a flat document font size.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_partially_bold_numbered_body_opening_line_is_not_a_heading() -> None:
    spans = [
        _heading_line(1, 0, "1. OBJETO DO SEGURO"),
        *_mixed_prefix_line(
            1,
            1,
            "1.1 ",
            "Pelo presente Bilhete de seguro, a Seguradora garante o pagamento.",
        ),
    ]
    document = _document([_page(1, spans)])

    tree = segment_document(document)

    assert tree.report.clause_count == 1
    node = tree.roots[0]
    assert node.numbering_label == "1"
    assert len(node.content_lines) == 1
    assert node.content_lines[0].startswith("1.1")


# ---------------------------------------------------------------------------
# 9. Numbering restart produces new top-level siblings (Bradesco bundle
#    pattern), with no cross-contamination between the two subtrees.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_numbering_restart_creates_independent_top_level_siblings() -> None:
    document = _document(
        [
            _page(
                1,
                [
                    _heading_line(1, 0, "1. REBOQUE APÓS ACIDENTE"),
                    _heading_line(1, 1, "1.1 Limites"),
                    _heading_line(1, 2, "2. CARRO RESERVA"),
                ],
            ),
            _page(
                2,
                [
                    _heading_line(2, 0, "1. MOTORISTA ANJO"),
                    _heading_line(2, 1, "1.1 Condições"),
                ],
            ),
        ]
    )

    tree = segment_document(document)

    assert [c.title for c in tree.roots] == [
        "1. REBOQUE APÓS ACIDENTE",
        "2. CARRO RESERVA",
        "1. MOTORISTA ANJO",
    ]
    first_bundle_sub = next(
        c for c in tree.all_clauses if c.parent_id == tree.roots[0].clause_id
    )
    second_bundle_sub = next(
        c for c in tree.all_clauses if c.parent_id == tree.roots[2].clause_id
    )
    assert first_bundle_sub.title == "1.1 Limites"
    assert second_bundle_sub.title == "1.1 Condições"


# ---------------------------------------------------------------------------
# 10. Depth-skip anomaly is clamped, not crashed, and warned about.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_depth_skip_anomaly_is_clamped_and_warned() -> None:
    document = _document(
        [
            _page(
                1,
                [
                    _heading_line(1, 0, "1. OBJETO DO SEGURO"),
                    _heading_line(1, 1, "2.1.5 Indenização Integral"),
                ],
            )
        ]
    )

    tree = segment_document(document)

    skipped = _find(tree, "2.1.5")
    assert skipped.depth == 2
    assert skipped.is_depth_anomaly is True
    assert skipped.parent_id == tree.roots[0].clause_id
    assert any(w.kind == "depth_anomaly" for w in tree.report.warnings)


@pytest.mark.unit
def test_depth_anomaly_does_not_become_false_floor_for_later_siblings() -> None:
    """[M1-04c] doc 20 samples #26/#27 (also confirmed reachable in 24/28
    text-mode corpus documents during investigation, not OCR-specific --
    this fixture is deliberately text-mode, answering the DoD's ask to
    confirm that). A bracket-numbered UNNUMBERED_PART section ("2)
    DEFINIÇÕES", depth 0) followed directly by its decimal children ("2.1"
    onward, natural depth 2) triggers a depth-skip anomaly on "2.1" alone
    (an intermediate "2." level was never a heading of its own). The bug:
    the clamped depth used to become "2.1"'s permanent depth, and later
    siblings ("2.2" .. "2.11") popped the stack by comparing against that
    same clamped value, nesting under "2.1" instead of popping past it.
    Tracking natural_depth (unclamped) separately from depth (tree
    position) fixes this -- every "2.x" must be a direct child of the
    root, not nested under "2.1", and only one depth_anomaly warning
    should fire (for "2.1" itself, not cascading to every sibling)."""
    labels = [f"2.{n}" for n in range(1, 12)]  # 2.1 .. 2.11 -- >= 10 per the DoD.
    document = _document(
        [
            _page(
                1,
                [_heading_line(1, 0, "2) DEFINIÇÕES")]
                + [
                    _heading_line(1, index + 1, f"{label} Termo {index}")
                    for index, label in enumerate(labels)
                ],
            )
        ]
    )

    tree = segment_document(document)

    root = tree.roots[0]
    assert root.title == "2) DEFINIÇÕES"
    children = [c for c in tree.all_clauses if c.parent_id == root.clause_id]
    assert [c.numbering_label for c in children] == labels
    depth_anomalies = [w for w in tree.report.warnings if w.kind == "depth_anomaly"]
    assert len(depth_anomalies) == 1


# ---------------------------------------------------------------------------
# 11. Orphan text before the first heading
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_text_before_first_heading_is_orphan() -> None:
    document = _document(
        [
            _page(
                1,
                [
                    _body_line(1, 0, "Texto solto sem cabeçalho."),
                    _heading_line(1, 1, "1. OBJETO DO SEGURO"),
                    _body_line(1, 2, "Corpo normal."),
                ],
            )
        ]
    )

    tree = segment_document(document)

    assert tree.report.orphan_char_count == len("Texto solto sem cabeçalho.")
    assert tree.report.orphan_ratio > 0
    node = tree.roots[0]
    assert node.content_lines == ("Corpo normal.",)


# ---------------------------------------------------------------------------
# 12. segment_document never raises, even with a very high orphan ratio --
#     OrphanTextExceedsThresholdError is a script-level concern (see
#     scripts/build_clause_tree.py), not raised by the pure use case.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_segment_document_never_raises_on_high_orphan_ratio() -> None:
    document = _document(
        [_page(1, [_body_line(1, i, f"Linha órfã {i}.") for i in range(20)])]
    )

    tree = segment_document(document)

    assert tree.report.orphan_ratio == 1.0
    assert tree.report.clause_count == 0


@pytest.mark.unit
def test_orphan_text_exceeds_threshold_carries_useful_fields() -> None:
    error = OrphanTextExceedsThresholdError(
        document_id="9",
        filename="f.pdf",
        orphan_ratio=0.42,
        threshold=0.15,
        clause_count=3,
    )

    assert error.document_id == "9"
    assert error.orphan_ratio == 0.42
    assert "0.420" in str(error)
    assert "0.150" in str(error)


# ---------------------------------------------------------------------------
# 13. Multi-column reflow: two independent numbering sequences running in
#     parallel columns must not interleave.
# ---------------------------------------------------------------------------


def _fixed_width_span(
    page_number: int, line_id: int, text: str, *, x0: float, x1: float, bold: bool
) -> ExtractedSpan:
    """Like [_span], but with an explicit bbox width independent of text
    length -- needed so the two synthetic columns land in predictable x0
    bands regardless of how long their sample text happens to be."""
    return ExtractedSpan(
        document_id="d1",
        page_number=page_number,
        line_id=line_id,
        order=line_id,
        bbox=(x0, 100.0, x1, 110.0),
        font_size=11.0,
        font_name="Test-BoldMT" if bold else "TestMT",
        text=text,
    )


@pytest.mark.unit
def test_multi_column_page_reflows_left_column_then_right_column() -> None:
    left_raw = [
        _fixed_width_span(1, 0, "2.44 Vigência do Seguro", x0=50.0, x1=250.0, bold=True)
    ] + [
        _fixed_width_span(1, i, f"Corpo esquerdo {i}", x0=50.0, x1=250.0, bold=False)
        for i in range(1, 14)
    ]
    right_raw = [
        _fixed_width_span(
            1, 100, "5.1 Despesas de salvamento", x0=290.0, x1=490.0, bold=True
        )
    ] + [
        _fixed_width_span(
            1, 100 + i, f"Corpo direito {i}", x0=290.0, x1=490.0, bold=False
        )
        for i in range(1, 14)
    ]
    # Fix y0 so both columns share the same vertical extent (genuinely
    # concurrent columns), then interleave in raw span order -- what a
    # multi-column PDF page's natural span order looks like before reflow.
    left = [_with_y0(span, y0=20.0 * i) for i, span in enumerate(left_raw)]
    right = [_with_y0(span, y0=20.0 * i) for i, span in enumerate(right_raw)]
    interleaved = [span for pair in zip(left, right, strict=True) for span in pair]
    document = _document([_page(1, interleaved)])

    tree = segment_document(document)

    titles = [c.title for c in tree.all_clauses]
    assert titles.index("2.44 Vigência do Seguro") < titles.index(
        "5.1 Despesas de salvamento"
    )
    assert any(w.kind == "multi_column_reflow" for w in tree.report.warnings)

    left_node = _find(tree, "2.44")
    right_node = _find(tree, "5.1")
    assert all(line.startswith("Corpo esquerdo") for line in left_node.content_lines)
    assert all(line.startswith("Corpo direito") for line in right_node.content_lines)


# ---------------------------------------------------------------------------
# 14/15. Multi-column false-positive guards: a table, and a single justified
#         column whose line-final word lands far to the right.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_table_like_page_does_not_trigger_multi_column_reflow() -> None:
    # Short-lived grid: many distinct x0 columns, but under the minimum line
    # count per band and with no sustained vertical overlap of prose.
    spans = [
        _body_line(1, i, f"Coluna {i}", x0=50.0 + (i % 4) * 120.0) for i in range(8)
    ]
    document = _document([_page(1, spans)])

    _, reflowed = _ordered_page_lines_for_test(document.pages[0])
    assert reflowed is False


@pytest.mark.unit
def test_justified_text_line_final_word_does_not_trigger_multi_column_reflow() -> None:
    # A single logical line (same line_id) whose last word is a separate
    # span landing far to the right, as PyMuPDF emits for justified text.
    spans = []
    for i in range(14):
        spans.append(
            _span(
                page_number=1, line_id=i, order=0, text=f"Início da linha {i} ", x0=50.0
            )
        )
        spans.append(_span(page_number=1, line_id=i, order=1, text="final", x0=480.0))
    document = _document([_page(1, spans)])

    _, reflowed = _ordered_page_lines_for_test(document.pages[0])
    assert reflowed is False


def _ordered_page_lines_for_test(page: ExtractedPage) -> tuple[list[Any], bool]:
    from application.use_cases.clause_segmentation import _ordered_page_lines

    return _ordered_page_lines(
        page, lambda font_name, font_size: "bold" in font_name.lower()
    )


# ---------------------------------------------------------------------------
# OCR path: pattern-only detection, no bold/position signal available.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_ocr_page_detects_headings_by_pattern_only() -> None:
    document = _document(
        [
            _ocr_page(
                1,
                "CONDIÇÕES CONTRATUAIS\n\n1. OBJETO DO SEGURO\n"
                "Texto do objeto do seguro.\n1.1 Âmbito\nTexto do âmbito.\n",
            )
        ]
    )

    tree = segment_document(document)

    assert tree.report.extraction_mode == "ocr_required"
    assert any(w.kind == "ocr_relaxed_mode" for w in tree.report.warnings)
    assert tree.report.clause_count >= 2


# ---------------------------------------------------------------------------
# clause_id/path determinism [M1-07]: same input produces the same id, and a
# structural edit confined to one branch never shifts the id of a clause in
# an unrelated branch -- see [application.use_cases.clause_segmentation.
# _slugify] and the sibling-segment-counting scheme in [segment_document].
# ---------------------------------------------------------------------------


def _coverages_document() -> ExtractedDocument:
    return _document(
        [
            _page(
                1,
                [
                    _heading_line(1, 1, "1. OBJETO DO SEGURO"),
                    _body_line(1, 2, "Corpo do objeto."),
                    _heading_line(1, 3, "2. COBERTURAS"),
                    _heading_line(1, 4, "2.1 Cobertura Basica"),
                    _body_line(1, 5, "Texto da cobertura basica."),
                ],
            )
        ]
    )


@pytest.mark.unit
def test_clause_id_deterministic_across_two_runs() -> None:
    document = _coverages_document()

    first = segment_document(document)
    second = segment_document(document)

    assert [c.clause_id for c in first.all_clauses] == [
        c.clause_id for c in second.all_clauses
    ]
    assert [c.path for c in first.all_clauses] == [c.path for c in second.all_clauses]


@pytest.mark.unit
def test_clause_id_stable_when_unrelated_clause_inserted() -> None:
    """Inserting a clause under "1" must not shift the id of "2"/"2.1"."""
    baseline = segment_document(_coverages_document())

    with_insertion = segment_document(
        _document(
            [
                _page(
                    1,
                    [
                        _heading_line(1, 1, "1. OBJETO DO SEGURO"),
                        _body_line(1, 2, "Corpo do objeto."),
                        _heading_line(1, 3, "1.1 Definicoes Extras"),
                        _body_line(1, 4, "Texto das definicoes extras."),
                        _heading_line(1, 5, "2. COBERTURAS"),
                        _heading_line(1, 6, "2.1 Cobertura Basica"),
                        _body_line(1, 7, "Texto da cobertura basica."),
                    ],
                )
            ]
        )
    )

    assert with_insertion.report.clause_count == baseline.report.clause_count + 1
    assert _find(with_insertion, "2").clause_id == _find(baseline, "2").clause_id
    assert _find(with_insertion, "2.1").clause_id == _find(baseline, "2.1").clause_id
    assert _find(with_insertion, "2").path == _find(baseline, "2").path


# ---------------------------------------------------------------------------
# [M1-04b]: recurring UNNUMBERED_PART candidates are demoted to content, not
# roots -- doc 10's repeated benefits-tier labels (e.g. "NÃO ESTÃO
# COBERTOS") are structurally identical to a genuine part title otherwise.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_recurring_bold_caps_line_is_demoted_to_content_not_a_root() -> None:
    document = _document(
        [
            _page(
                1,
                [
                    _heading_line(1, 0, "1. RISCOS EXCLUÍDOS"),
                    _body_line(1, 1, "Texto de exclusão."),
                    _heading_line(1, 2, "NÃO ESTÃO COBERTOS"),
                ],
            ),
            _page(
                2,
                [
                    _heading_line(2, 0, "2. OUTRAS DISPOSIÇÕES"),
                    _body_line(2, 1, "Mais texto."),
                    _heading_line(2, 2, "NÃO ESTÃO COBERTOS"),
                ],
            ),
        ]
    )

    tree = segment_document(document)

    assert [c.title for c in tree.roots] == [
        "1. RISCOS EXCLUÍDOS",
        "2. OUTRAS DISPOSIÇÕES",
    ]
    assert "NÃO ESTÃO COBERTOS" in tree.roots[0].content_lines
    assert "NÃO ESTÃO COBERTOS" in tree.roots[1].content_lines


@pytest.mark.unit
def test_recurring_title_with_accent_and_spelling_drift_is_still_grouped() -> None:
    """doc 10's real extraction inconsistency: the same noise item surfaces
    once as "ENVIO DE TÁXI" and again as "ENVIO DE TAXI" -- both must be
    recognized as the same recurring group and demoted together."""
    document = _document(
        [
            _page(
                1,
                [
                    _heading_line(1, 0, "1. SERVIÇOS INCLUSOS"),
                    _body_line(1, 1, "Texto de serviços."),
                    _heading_line(1, 2, "ENVIO DE TÁXI"),
                ],
            ),
            _page(
                2,
                [
                    _heading_line(2, 0, "2. OUTROS SERVIÇOS"),
                    _body_line(2, 1, "Mais texto."),
                    _heading_line(2, 2, "ENVIO DE TAXI"),
                ],
            ),
        ]
    )

    tree = segment_document(document)

    assert [c.title for c in tree.roots] == [
        "1. SERVIÇOS INCLUSOS",
        "2. OUTROS SERVIÇOS",
    ]
    assert "ENVIO DE TÁXI" in tree.roots[0].content_lines
    assert "ENVIO DE TAXI" in tree.roots[1].content_lines


@pytest.mark.unit
def test_recurring_title_always_followed_by_restart_is_kept_as_bundle_roots() -> None:
    """doc 16's real case: "CONDIÇÕES ESPECIAIS" is reused for 3 distinct
    coverage variants, each immediately followed by its own "1." restart --
    unlike doc 10's noise, every occurrence qualifies, so all must survive
    as independent, high-confidence bundle roots."""
    document = _document(
        [
            _page(
                1,
                [
                    _heading_line(1, 0, "CONDIÇÕES ESPECIAIS"),
                    _heading_line(1, 1, "1. RISCOS COBERTOS"),
                    _body_line(1, 2, "Texto 1."),
                ],
            ),
            _page(
                2,
                [
                    _heading_line(2, 0, "CONDIÇÕES ESPECIAIS"),
                    _heading_line(2, 1, "1. RISCOS COBERTOS"),
                    _body_line(2, 2, "Texto 2."),
                ],
            ),
        ]
    )

    tree = segment_document(document)

    assert len(tree.roots) == 2
    assert all(c.title == "CONDIÇÕES ESPECIAIS" for c in tree.roots)
    assert all(c.bundle_confidence == "high" for c in tree.roots)


@pytest.mark.unit
def test_recurring_title_with_partial_restart_is_demoted_in_full() -> None:
    """doc 10's real case: "ARGENTINA, PARAGUAI, URUGUAI E CHILE." recurs,
    and only one of its two occurrences happens to precede an unrelated
    "1." restart -- the all-or-nothing rule must demote it entirely,
    including the occurrence that individually looked like a real restart."""
    document = _document(
        [
            _page(
                1,
                [
                    _heading_line(1, 0, "ARGENTINA, PARAGUAI, URUGUAI E CHILE."),
                    _heading_line(1, 1, "1. COBERTURA ESPECIAL"),
                    _body_line(1, 2, "Texto."),
                ],
            ),
            _page(
                2,
                [
                    _heading_line(2, 0, "2. OUTRA SEÇÃO"),
                    _body_line(2, 1, "Corpo da seção."),
                    _heading_line(2, 2, "ARGENTINA, PARAGUAI, URUGUAI E CHILE."),
                    _body_line(2, 3, "Outro texto."),
                ],
            ),
        ]
    )

    tree = segment_document(document)

    assert [c.title for c in tree.roots] == [
        "1. COBERTURA ESPECIAL",
        "2. OUTRA SEÇÃO",
    ]
    assert "ARGENTINA, PARAGUAI, URUGUAI E CHILE." in tree.roots[1].content_lines
    # The first occurrence was demoted before any clause was open, so it
    # landed as orphan text rather than being silently dropped or promoted.
    assert tree.report.orphan_char_count > 0


@pytest.mark.unit
def test_part_root_promoted_when_only_child_restarts_at_non_one_label() -> None:
    """doc 10's real case: "MOTOCICLETAS"'s own assistance-list items never
    surface as a "1."-labeled child (the first real child is "10."), so the
    relaxed [M1-06] check must promote it on the presence of any numbered
    child, not specifically a restart at "1"."""
    document = _document(
        [
            _page(
                1,
                [
                    _heading_line(1, 0, "ASSISTÊNCIA MOTOCICLETAS"),
                    _heading_line(1, 1, "10. ITEM DEZ"),
                    _body_line(1, 2, "Texto item dez."),
                ],
            ),
            _page(
                2,
                [
                    _heading_line(2, 0, "DISPOSIÇÕES GERAIS"),
                    _heading_line(2, 1, "1. OBJETO"),
                    _body_line(2, 2, "Texto objeto."),
                ],
            ),
        ]
    )

    tree = segment_document(document)

    motocicletas = next(c for c in tree.roots if c.title == "ASSISTÊNCIA MOTOCICLETAS")
    assert motocicletas.bundle_confidence == "high"
    assert motocicletas.bundle_section == "ASSISTÊNCIA MOTOCICLETAS"


@pytest.mark.unit
def test_part_root_with_zero_numbered_children_is_not_promoted() -> None:
    """doc 10's real case: "RESUMO DE COBERTURAS DA ASSISTÊNCIA À VEÍCULOS
    DE CARGA" is a genuine, correctly-recognized root (DoD bullet 3) but is
    a pure coverage-comparison table with no numbered sub-clauses -- it
    must stay a root without being promoted to bundle_confidence="high"."""
    document = _document(
        [
            _page(
                1,
                [
                    _heading_line(1, 0, "DISPOSIÇÕES GERAIS"),
                    _heading_line(1, 1, "1. OBJETO"),
                    _body_line(1, 2, "Texto objeto."),
                ],
            ),
            _page(
                2,
                [
                    _heading_line(
                        2, 0, "RESUMO DE COBERTURAS DA ASSISTÊNCIA À VEÍCULOS DE CARGA"
                    ),
                    _body_line(2, 1, "Tabela de coberturas."),
                ],
            ),
            _page(
                3,
                [
                    _heading_line(3, 0, "ASSISTÊNCIA MOTOCICLETAS"),
                    _heading_line(3, 1, "10. ITEM DEZ"),
                    _body_line(3, 2, "Texto item dez."),
                ],
            ),
        ]
    )

    tree = segment_document(document)

    carga = next(c for c in tree.roots if "CARGA" in c.title)
    assert carga.bundle_confidence == "low"
    assert carga.bundle_section is None


# ---------------------------------------------------------------------------
# 16. Oversized-clause safeguard
# ---------------------------------------------------------------------------


def _sized_clause(
    clause_id: str, *, page_start: int, page_end: int, content_lines: tuple[str, ...]
) -> Clause:
    return Clause(
        document_id="d1",
        clause_id=clause_id,
        path=clause_id,
        numbering_label="1",
        title="Title",
        convention=HeadingConvention.NUMBERED_DECIMAL,
        depth=1,
        parent_id=None,
        child_ids=(),
        content_lines=content_lines,
        page_start=page_start,
        page_end=page_end,
    )


@pytest.mark.unit
def test_find_oversized_clauses_flags_page_span_and_char_count_outliers() -> None:
    """doc 13 sample #16: a page-span or char-count ceiling is the
    loud-failure safeguard for an undetected-heading merge like "RISCOS
    EXCLUÍDOS" absorbing 20 pages / 41,000+ characters. Isolated fixture
    per the DoD -- directly constructs [Clause] objects, no PDF or
    [segment_document] call needed to exercise this pure helper."""
    in_bounds = _sized_clause(
        "d1:1", page_start=1, page_end=2, content_lines=("short text",)
    )
    oversized_by_pages = _sized_clause(
        "d1:2", page_start=1, page_end=21, content_lines=("short text",)
    )
    oversized_by_chars = _sized_clause(
        "d1:3", page_start=1, page_end=1, content_lines=("x" * 20000,)
    )

    oversized = find_oversized_clauses(
        [in_bounds, oversized_by_pages, oversized_by_chars],
        max_page_span=10,
        max_char_count=15000,
    )

    assert {c.clause_id for c in oversized} == {"d1:2", "d1:3"}


# ---------------------------------------------------------------------------
# 17. Per-line page attribution (M1-04d)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_content_line_pages_parallels_content_lines_across_a_page_break() -> None:
    """[M1-04d] needs to know which page each content line came from, to
    apply a vision-proposed boundary correction without re-running heading
    detection -- see [application.use_cases.boundary_escalation]."""
    spans = [
        _heading_line(1, 0, "1. Riscos Cobertos"),
        _body_line(1, 1, "Primeira linha de corpo na pagina 1."),
        _body_line(2, 0, "Segunda linha de corpo na pagina 2."),
        _body_line(2, 1, "Terceira linha de corpo na pagina 2."),
    ]
    document = _document([_page(1, spans[0:2]), _page(2, spans[2:4])])

    tree = segment_document(document)

    clause = _find(tree, "1")
    assert clause.content_lines == (
        "Primeira linha de corpo na pagina 1.",
        "Segunda linha de corpo na pagina 2.",
        "Terceira linha de corpo na pagina 2.",
    )
    assert clause.content_line_pages == (1, 2, 2)
    assert len(clause.content_line_pages) == len(clause.content_lines)


@pytest.mark.unit
def test_non_bold_numbering_successor_opens_its_own_clause() -> None:
    """[M1-08c] doc 8 p52: siblings typeset inconsistently still split.

    "6.3" is set bold (bold_fraction 0.99) but its own siblings "6.4"/"6.5"
    sit in the body font (0.03), so the bold gate in [_detect_heading]
    swallows them into 6.3's content -- the adjacent-sibling merge that was
    the largest remaining boundary-failure cluster in both [M1-08b] and
    [M1-08c]. Numbering continuity is evidence independent of typesetting.
    """
    lines_spec = [
        (_heading_line(1, 0, "6. INDENIZAÇÃO"), 100.0),
        (_body_line(1, 1, "Texto de corpo comum um."), 115.0),
        (_body_line(1, 2, "Texto de corpo comum dois."), 130.0),
        (_body_line(1, 3, "Texto de corpo comum tres."), 145.0),
        (_body_line(1, 4, "Texto de corpo comum quatro."), 160.0),
        (_body_line(1, 5, "Texto de corpo comum cinco."), 175.0),
        # 30pt gap, bold: detected by the ordinary numbered gate.
        (_heading_line(1, 6, "6.3 As condições mencionadas acima"), 205.0),
        (_body_line(1, 7, "na modalidade Valor Determinado."), 220.0),
        # 30pt gap, NOT bold: only numbering continuity identifies it.
        (_body_line(1, 8, "6.4 Correrão por conta da Seguradora"), 250.0),
        (_body_line(1, 9, "as despesas de contenção e salvamento."), 265.0),
        # 30pt gap, NOT bold: successor of 6.4, same fallback.
        (_body_line(1, 10, "6.5 Correrão por conta da Seguradora"), 295.0),
        (_body_line(1, 11, "os valores referentes aos danos materiais."), 310.0),
    ]
    spans = [_with_y0(span, y0=y0) for span, y0 in lines_spec]

    tree = segment_document(_document([_page(1, spans)]))

    labels = {c.numbering_label for c in tree.all_clauses}
    assert {"6.3", "6.4", "6.5"} <= labels

    clause_63 = _find(tree, "6.3")
    assert clause_63.content_lines == ("na modalidade Valor Determinado.",)
    assert _find(tree, "6.4").content_lines == (
        "as despesas de contenção e salvamento.",
    )
    # 6.4/6.5 are siblings of 6.3, not children of it.
    assert _find(tree, "6.4").parent_id == clause_63.parent_id
    assert _find(tree, "6.5").parent_id == clause_63.parent_id


@pytest.mark.unit
def test_non_successor_numbering_stays_content() -> None:
    """The fallback is continuity-gated: an unrelated label is not a heading.

    Without this, any body line opening with a decimal token would be
    promoted, which is exactly the over-splitting the bold gate exists to
    prevent in the first place.
    """
    lines_spec = [
        (_heading_line(1, 0, "6. INDENIZAÇÃO"), 100.0),
        (_body_line(1, 1, "Texto de corpo comum um."), 115.0),
        (_body_line(1, 2, "Texto de corpo comum dois."), 130.0),
        (_body_line(1, 3, "Texto de corpo comum tres."), 145.0),
        (_body_line(1, 4, "Texto de corpo comum quatro."), 160.0),
        (_heading_line(1, 5, "6.3 As condições mencionadas acima"), 190.0),
        # 30pt gap but "6.9" does not follow "6.3" -- stays content.
        (_body_line(1, 6, "6.9 conforme o item citado anteriormente."), 220.0),
    ]
    spans = [_with_y0(span, y0=y0) for span, y0 in lines_spec]

    tree = segment_document(_document([_page(1, spans)]))

    assert "6.9" not in {c.numbering_label for c in tree.all_clauses}
    assert _find(tree, "6.3").content_lines == (
        "6.9 conforme o item citado anteriormente.",
    )


@pytest.mark.unit
def test_numbering_successor_at_ordinary_line_pitch_still_opens_a_clause() -> None:
    """[M1-08c] docs 17/24: spacing carries no heading signal in some documents.

    Both set their numbered sub-clauses at exactly the ordinary line pitch
    (15.72-15.84pt against a 15.8pt modal gap), so gating the
    numbering-continuity fallback on [_has_heading_position_signal] left
    those merges unfixed. The gate is deliberately not applied here -- the
    same way the main loop never applies it to bold numbered headings --
    leaving exact-successor matching as the discriminator.

    Accepted tradeoff: a wrapped body line beginning with precisely the
    successor label would now be promoted to a heading. That requires the
    wrap to land on the one token that continues the sequence, which the
    corpus evidence behind [M1-08c] did not show occurring, while the
    merges this admits were its largest boundary-failure cluster.
    """
    lines_spec = [
        (_heading_line(1, 0, "6. INDENIZAÇÃO"), 100.0),
        (_body_line(1, 1, "Texto de corpo comum um."), 115.0),
        (_body_line(1, 2, "Texto de corpo comum dois."), 130.0),
        (_body_line(1, 3, "Texto de corpo comum tres."), 145.0),
        (_body_line(1, 4, "Texto de corpo comum quatro."), 160.0),
        (_heading_line(1, 5, "6.3 As condições mencionadas acima"), 190.0),
        # Ordinary 15pt pitch -- no extra spacing, as in docs 17 and 24.
        (_body_line(1, 6, "6.4 Correrão por conta da Seguradora"), 205.0),
    ]
    spans = [_with_y0(span, y0=y0) for span, y0 in lines_spec]

    tree = segment_document(_document([_page(1, spans)]))

    assert "6.4" in {c.numbering_label for c in tree.all_clauses}
    assert _find(tree, "6.3").content_lines == ()
