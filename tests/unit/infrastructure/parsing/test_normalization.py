import pytest

from infrastructure.parsing.normalization import (
    apply_nfkc,
    normalize_nbsp,
    normalize_quotes,
    normalize_text,
    should_rejoin,
    strip_soft_hyphens,
)


@pytest.mark.unit
def test_apply_nfkc_folds_ligatures() -> None:
    assert apply_nfkc("eﬁciência") == "eficiência"


@pytest.mark.unit
def test_strip_soft_hyphens_removes_soft_hyphen_unconditionally() -> None:
    assert strip_soft_hyphens("infor\xadmação") == "informação"


@pytest.mark.unit
def test_strip_soft_hyphens_leaves_real_hyphens_alone() -> None:
    assert strip_soft_hyphens("sócio-cultural") == "sócio-cultural"


@pytest.mark.unit
def test_normalize_nbsp_replaces_with_regular_space() -> None:
    assert normalize_nbsp("R$\xa010,00") == "R$ 10,00"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("“Cláusula”", '"Cláusula"'),
        ("‘risco’", "'risco'"),
        ("«franquia»", '"franquia"'),
        ("„franquia“", '"franquia"'),
    ],
)
def test_normalize_quotes_maps_variants_to_straight_quotes(
    raw: str, expected: str
) -> None:
    assert normalize_quotes(raw) == expected


@pytest.mark.unit
def test_normalize_text_runs_the_full_pipeline() -> None:
    raw = "“Eﬁciência\xad”\xa0e\xa0seguro"

    result = normalize_text(raw)

    assert result == '"Eficiência" e seguro'


@pytest.mark.unit
def test_should_rejoin_merges_a_word_wrapped_at_the_line_break() -> None:
    # "informa-" / "ção" -- lowercase continuation, classic line-wrap.
    assert should_rejoin("O prazo para informa-", "ção do sinistro") is True


@pytest.mark.unit
def test_should_rejoin_does_not_merge_uppercase_continuation() -> None:
    # "RCF-" / "V" -- observed in the real corpus (Bradesco RCF-A policy,
    # 2014 filing): the hyphen is part of "RCF-V", not a wrap artifact.
    assert should_rejoin("cobertura de RCF-", "V, estão garantidos") is False


@pytest.mark.unit
def test_should_rejoin_does_not_merge_when_no_trailing_hyphen() -> None:
    assert should_rejoin("O prazo é de trinta dias", "para avisar o sinistro") is False


@pytest.mark.unit
def test_should_rejoin_cannot_distinguish_a_genuine_compound() -> None:
    # Known, accepted limitation: "sócio-" / "cultural" is a genuine
    # hyphenated compound, but is structurally identical to a line-wrap
    # (lowercase before and after the hyphen), so this also returns True.
    # Disambiguating would need a Portuguese dictionary, out of scope here.
    assert should_rejoin("o benefício sócio-", "cultural oferecido") is True
