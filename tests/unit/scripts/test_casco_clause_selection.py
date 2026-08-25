"""Tests for deterministic CASCO source-clause selection [M2-02].

Each group covers one selection hazard the module guards against; no LLM
calls.
"""

import pytest
from scripts.casco_clause_selection import (
    MAX_QUESTIONS_PER_DOC,
    VOCAB_PATTERNS,
    ClauseCandidate,
    build_ancestor_titles,
    build_duplicate_text_index,
    classify_scope,
    coverage_with_exclusion_gap,
    find_exclusion_reasons,
    pick_slots,
    question_is_indemnity_basis,
    question_scope_flag,
    question_self_reference_flag,
    score_question_scenarios,
    score_question_vocabulary,
    select_indemnity_basis_documents,
)

from infrastructure.parsing.clause_schema import ParsedClauseRecord


def make_record(**overrides: object) -> ParsedClauseRecord:
    """Build a valid CASCO ParsedClauseRecord, overridable per test."""
    fields: dict[str, object] = {
        "schema_version": "v1",
        "clause_id": "1:14-riscos-cobertos",
        "document_id": "1",
        "parent_id": None,
        "path": "14-riscos-cobertos",
        "title": "14. RISCOS COBERTOS",
        "text": "Garante os prejuízos decorrentes de colisão do veículo segurado.",
        "clause_type": "coverage",
        "type_source": "rule",
        "confidence": None,
        "bundle_section": None,
        "page_start": 1,
        "page_end": 1,
        "source": "text",
        "boundary_source": None,
        "susep_process": "123",
        "insurer": "Porto Seguro",
        "cnpj": "00.000.000/0001-00",
        "product_line": "CASCO",
        "indemnity_regime": "VD",
        "filing_year": "2024",
    }
    fields.update(overrides)
    return ParsedClauseRecord.model_validate(fields)


def make_candidate(**overrides: object) -> ClauseCandidate:
    """Build a selectable ClauseCandidate, overridable per test."""
    fields: dict[str, object] = {
        "clause": make_record(),
        "vocab_hits": frozenset(),
        "is_indemnity_basis": False,
        "scope": "casco",
        "exclusion_reasons": (),
        "twin_ids": frozenset(),
    }
    fields.update(overrides)
    return ClauseCandidate(**fields)  # type: ignore[arg-type]


# --- vocabulary scoring: measured on the question, not the clause ---------


@pytest.mark.unit
def test_score_question_vocabulary_reads_the_question_text() -> None:
    hits = score_question_vocabulary("O incêndio do veículo está coberto?")
    assert hits == frozenset({"incendio"})


@pytest.mark.unit
def test_perda_total_does_not_match_app_disability_wording() -> None:
    assert score_question_vocabulary("perda total da visão de um olho") == frozenset()
    assert score_question_vocabulary("perda total do uso de um membro") == frozenset()


@pytest.mark.unit
def test_perda_total_matches_vehicle_total_loss_and_its_synonym() -> None:
    assert "perda_total" in score_question_vocabulary("houve perda total do veículo?")
    assert "perda_total" in score_question_vocabulary("cabe indenização integral?")


@pytest.mark.unit
def test_question_is_indemnity_basis_reads_the_question_text() -> None:
    assert question_is_indemnity_basis("Como se apura o valor de mercado referenciado?")
    assert not question_is_indemnity_basis("Qual a diferença entre furto e roubo?")


# --- scope classification -------------------------------------------------


@pytest.mark.unit
def test_classify_scope_flags_residential_from_ancestor_title() -> None:
    record = make_record(title="1. Coberturas", text="Cobre o conteúdo.")
    scope = classify_scope(
        record, ("CONDIÇÕES ESPECIAIS - COBERTURAS PARA A RESIDÊNCIA",)
    )
    assert scope == "fora_escopo:residencial"


@pytest.mark.unit
def test_classify_scope_flags_personal_accident_in_a_small_leaf_clause() -> None:
    record = make_record(
        title="1. Objeto", text="Garante o pagamento por invalidez permanente."
    )
    assert classify_scope(record, ()) == "fora_escopo:app_acidentes_pessoais"


@pytest.mark.unit
def test_classify_scope_keeps_a_glossary_in_scope_despite_its_body() -> None:
    """A bundled glossary defines every product's terms; that is not misuse.

    Excluding it marked 13 sound own-damage questions out of scope. The
    misuse the review actually objected to is a question that asks a glossary
    for a personal-accident definition, which `question_scope_flag` catches.
    """
    record = make_record(title="3. Definições", text="Acidente Pessoal é o evento...")
    assert classify_scope(record, (), child_count=36) == "casco"
    assert classify_scope(record, ()) == "casco"


@pytest.mark.unit
def test_classify_scope_keeps_a_combined_product_heading_in_scope() -> None:
    record = make_record(title="1. Objeto", text="Cobertura do veículo.")
    scope = classify_scope(
        record, ("CONDIÇÕES GERAIS DOS SEGUROS DE AUTOMÓVEL, RCF-V E APP",)
    )
    assert scope == "casco"


@pytest.mark.unit
def test_question_scope_flag_catches_an_out_of_scope_question() -> None:
    assert (
        question_scope_flag("O que caracteriza um Acidente Pessoal?")
        == "fora_escopo:app_acidentes_pessoais"
    )
    assert question_scope_flag("O que é considerado salvado do veículo?") == ""


@pytest.mark.unit
def test_classify_scope_flags_named_unmarked_injury_tables() -> None:
    record = make_record(clause_id="6:t-o-t-a-l", title="T O T A L", text="100%")
    assert classify_scope(record, ()) == "fora_escopo:app_acidentes_pessoais"


@pytest.mark.unit
def test_classify_scope_tags_assistance_without_excluding_it() -> None:
    record = make_record(title="3.2 REBOQUE", text="Serviço de reboque do veículo.")
    assert classify_scope(record, ()) == "periferico:assistencia"


@pytest.mark.unit
def test_classify_scope_keeps_third_party_liability_in_scope() -> None:
    """RCF is required M2-02 vocabulary, not a bundled foreign product."""
    record = make_record(
        title="12. RESPONSABILIDADE CIVIL FACULTATIVA",
        text="Garante danos causados a terceiros.",
    )
    assert classify_scope(record, ()) == "casco"


# --- degenerate-clause guard ---------------------------------------------


@pytest.mark.unit
def test_find_exclusion_reasons_flags_duplicate_text() -> None:
    reasons = find_exclusion_reasons(make_record(), frozenset({"9:1-4"}), 0)
    assert "duplicate_text" in reasons


@pytest.mark.unit
def test_find_exclusion_reasons_flags_title_split_mid_sentence() -> None:
    record = make_record(
        title=(
            "8.1.3.1 Uma vez que o seguro é mensal, a indenização será com "
            "base no valor determinado,"
        ),
        text="expresso em reais e definido no Bilhete de seguro vigente.",
    )
    assert "split_mid_sentence" in find_exclusion_reasons(record, frozenset(), 0)


@pytest.mark.unit
def test_find_exclusion_reasons_flags_spaced_letter_artifact_title() -> None:
    record = make_record(title="P A R C I A L MEMBROS SUPERIORES", text="x" * 60)
    assert "artifact_title" in find_exclusion_reasons(record, frozenset(), 0)


@pytest.mark.unit
def test_find_exclusion_reasons_accepts_an_ordinary_clause() -> None:
    assert find_exclusion_reasons(make_record(), frozenset(), 0) == ()


@pytest.mark.unit
def test_find_exclusion_reasons_prefers_child_over_thin_container() -> None:
    record = make_record(text="Ver subitens abaixo.")
    assert "container_prefer_child" in find_exclusion_reasons(record, frozenset(), 3)


# --- duplicate index ------------------------------------------------------


@pytest.mark.unit
def test_build_duplicate_text_index_groups_identical_text_in_one_document() -> None:
    shared = "A cobertura vigora exclusivamente nos sinistros por indenização integral."
    records = [
        make_record(clause_id="9:1-2", text=shared),
        make_record(clause_id="9:1-4", text=shared),
        make_record(clause_id="9:1-5", text=shared),
        make_record(
            clause_id="9:1-7", text="Texto diferente o suficiente para contar."
        ),
    ]
    twins = build_duplicate_text_index(records)
    assert twins["9:1-2"] == frozenset({"9:1-4", "9:1-5"})
    assert "9:1-7" not in twins


@pytest.mark.unit
def test_build_duplicate_text_index_does_not_pair_across_documents() -> None:
    shared = "Texto compartilhado entre dois documentos distintos, longo o bastante."
    records = [
        make_record(clause_id="1:a", document_id="1", text=shared),
        make_record(clause_id="2:a", document_id="2", text=shared),
    ]
    assert build_duplicate_text_index(records) == {}


# --- ancestor chain -------------------------------------------------------


@pytest.mark.unit
def test_build_ancestor_titles_walks_to_the_root() -> None:
    records = [
        make_record(clause_id="8:2", parent_id=None, title="2 DESPESAS COBERTAS"),
        make_record(
            clause_id="8:2/3", parent_id="8:2", title="3. ASSISTÊNCIA 24 HORAS"
        ),
        make_record(clause_id="8:2/3/3.2", parent_id="8:2/3", title="3.2 REBOQUE"),
    ]
    chains = build_ancestor_titles(records)
    assert chains["8:2/3/3.2"] == ("3. ASSISTÊNCIA 24 HORAS", "2 DESPESAS COBERTAS")
    assert chains["8:2"] == ()


# --- slot picking ---------------------------------------------------------


@pytest.mark.unit
def test_pick_slots_never_selects_an_out_of_scope_clause() -> None:
    out_of_scope = make_candidate(
        clause=make_record(clause_id="11:res"), scope="fora_escopo:residencial"
    )
    ok = make_candidate(clause=make_record(clause_id="11:ok"))
    slots = pick_slots(
        [out_of_scope, ok], dict.fromkeys(VOCAB_PATTERNS, 0), want_indemnity_basis=False
    )
    assert [c.clause.clause_id for c in slots] == ["11:ok"]


@pytest.mark.unit
def test_pick_slots_never_selects_a_degenerate_clause() -> None:
    degenerate = make_candidate(
        clause=make_record(clause_id="9:1-2"),
        exclusion_reasons=("duplicate_text",),
        twin_ids=frozenset({"9:1-4"}),
    )
    ok = make_candidate(clause=make_record(clause_id="9:2"))
    slots = pick_slots(
        [degenerate, ok], dict.fromkeys(VOCAB_PATTERNS, 0), want_indemnity_basis=False
    )
    assert [c.clause.clause_id for c in slots] == ["9:2"]


@pytest.mark.unit
def test_pick_slots_deprioritizes_assistance_so_it_cannot_dominate() -> None:
    assistance = make_candidate(
        clause=make_record(clause_id="12:assist"), scope="periferico:assistencia"
    )
    casco = make_candidate(clause=make_record(clause_id="12:casco"))
    slots = pick_slots(
        [assistance, casco],
        dict.fromkeys(VOCAB_PATTERNS, 0),
        want_indemnity_basis=False,
    )
    assert slots[0].clause.clause_id == "12:casco"


@pytest.mark.unit
def test_pick_slots_prefers_lowest_running_vocabulary_count() -> None:
    common = make_candidate(
        clause=make_record(clause_id="1:a"), vocab_hits=frozenset({"franquia"})
    )
    scarce = make_candidate(
        clause=make_record(clause_id="1:b"), vocab_hits=frozenset({"incendio"})
    )
    counts = dict.fromkeys(VOCAB_PATTERNS, 0)
    counts["franquia"] = 9
    slots = pick_slots([common, scarce], counts, want_indemnity_basis=False)
    assert slots[0].clause.clause_id == "1:b"


@pytest.mark.unit
def test_pick_slots_honours_limit_and_already_used_ids() -> None:
    candidates = [
        make_candidate(clause=make_record(clause_id=f"1:{i}")) for i in range(6)
    ]
    slots = pick_slots(
        candidates,
        dict.fromkeys(VOCAB_PATTERNS, 0),
        want_indemnity_basis=False,
        limit=2,
        already_used_ids=frozenset({"1:0"}),
    )
    assert len(slots) == 2
    assert "1:0" not in {c.clause.clause_id for c in slots}


@pytest.mark.unit
def test_pick_slots_caps_at_max_questions_per_doc() -> None:
    candidates = [
        make_candidate(clause=make_record(clause_id=f"1:{i}")) for i in range(20)
    ]
    slots = pick_slots(
        candidates, dict.fromkeys(VOCAB_PATTERNS, 0), want_indemnity_basis=False
    )
    assert len(slots) == MAX_QUESTIONS_PER_DOC


@pytest.mark.unit
def test_select_indemnity_basis_documents_spans_all_three_regimes() -> None:
    candidates_by_doc = {
        doc: [
            make_candidate(
                clause=make_record(clause_id=f"{doc}:b"), is_indemnity_basis=True
            )
        ]
        for doc in ("1", "2", "7", "8", "10", "11")
    }
    regimes = {
        "1": "VD",
        "2": "VD",
        "7": "VMR",
        "8": "VMR",
        "10": "VD+VMR",
        "11": "VD+VMR",
    }
    chosen = select_indemnity_basis_documents(candidates_by_doc, regimes)
    assert {regimes[d] for d in chosen} == {"VD", "VMR", "VD+VMR"}
    assert len(chosen) >= 5


@pytest.mark.unit
def test_select_indemnity_basis_documents_ignores_unselectable_clauses() -> None:
    candidates_by_doc = {
        "1": [
            make_candidate(
                clause=make_record(clause_id="1:b"),
                is_indemnity_basis=True,
                exclusion_reasons=("duplicate_text",),
            )
        ]
    }
    assert select_indemnity_basis_documents(candidates_by_doc, {"1": "VD"}) == set()


# --- round-3 gates: scenario depth, granularity, coverage+exclusion -------


@pytest.mark.unit
def test_scenario_ignores_a_coverage_package_listing() -> None:
    """Naming a package is not a claim (review: 2 of 3 incêndio hits were this)."""
    question = (
        "Se o veículo sofrer dano na cobertura Compreensiva "
        "(Colisão, Incêndio, Roubo/Furto e Alagamento), há indenização?"
    )
    assert score_question_scenarios(question, "direct_lookup") == frozenset()


@pytest.mark.unit
def test_scenario_counts_a_real_claim_situation() -> None:
    question = "O veículo pegou fogo após um curto-circuito; o incêndio é coberto?"
    assert "incendio" in score_question_scenarios(question, "direct_lookup")


@pytest.mark.unit
def test_scenario_never_counts_a_definition_question() -> None:
    question = "Se ocorrer colisão, o que a apólice define como colisão?"
    assert score_question_scenarios(question, "definition") == frozenset()


@pytest.mark.unit
def test_scenario_requires_a_situation_marker() -> None:
    assert (
        score_question_scenarios("Qual é a franquia?", "direct_lookup") == frozenset()
    )


@pytest.mark.unit
def test_question_self_reference_flag_catches_deste_documento() -> None:
    assert question_self_reference_flag("Na cobertura X deste documento, qual o valor?")
    assert question_self_reference_flag("Qual o prazo para avisar o sinistro?") == ""


@pytest.mark.unit
def test_glossary_container_with_children_is_not_anchorable() -> None:
    record = make_record(title="2. GLOSSÁRIO", text="x" * 9000)
    assert "glossary_container_prefer_entry" in find_exclusion_reasons(
        record, frozenset(), 40
    )


@pytest.mark.unit
def test_inline_glossary_without_children_stays_anchorable() -> None:
    record = make_record(title="GLOSSÁRIO", text="x" * 9000)
    assert "glossary_container_prefer_entry" not in find_exclusion_reasons(
        record, frozenset(), 0
    )


@pytest.mark.unit
def test_split_detection_catches_a_body_opening_with_a_parenthesis() -> None:
    record = make_record(
        title="1.1. Considera-se como veículo 0 km para os fins desta cobertur",
        text="(duzentos e setenta) dias, contados a partir da data de entrega.",
    )
    assert "split_mid_sentence" in find_exclusion_reasons(record, frozenset(), 0)


@pytest.mark.unit
def test_coverage_with_exclusion_gap_requires_both_sides() -> None:
    by_id = {
        "c": make_record(clause_id="c", clause_type="coverage"),
        "e": make_record(clause_id="e", clause_type="exclusion"),
        "f": make_record(clause_id="f", clause_type="condition"),
    }
    assert coverage_with_exclusion_gap(["c", "e"], by_id) == ""
    assert coverage_with_exclusion_gap(["e"], by_id) == "missing_coverage"
    assert coverage_with_exclusion_gap(["e", "f"], by_id) == "missing_coverage"
    assert coverage_with_exclusion_gap(["c"], by_id) == "missing_exclusion"
