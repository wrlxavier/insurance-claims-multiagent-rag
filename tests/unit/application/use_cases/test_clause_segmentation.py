from typing import Any

import pytest

from application.use_cases.clause_segmentation import segment_document
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

    return _ordered_page_lines(page, lambda font_name: "bold" in font_name.lower())


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
