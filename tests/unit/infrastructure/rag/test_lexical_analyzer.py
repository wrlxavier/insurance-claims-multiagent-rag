"""Portuguese analyzer for the BM25 lexical retriever -- [M3-03]."""

import pytest

from infrastructure.rag.lexical_analyzer import (
    TextAnalyzer,
    build_analyzer,
    fold_accents,
    normalize,
    normalized_tokens,
    tokenize,
)


class RecordingStemmer:
    """A fake stemmer: records every word it is asked to stem, chops one char."""

    def __init__(self) -> None:
        self.seen: list[str] = []

    def stemWord(self, word: str) -> str:  # noqa: N802 -- matches the real API
        self.seen.append(word)
        return word[:-1] if len(word) > 3 else word


def _analyzer(protected: set[str]) -> tuple[TextAnalyzer, RecordingStemmer]:
    stemmer = RecordingStemmer()
    return TextAnalyzer(protected_tokens=frozenset(protected), stemmer=stemmer), stemmer


@pytest.mark.unit
def test_normalize_lowercases_and_keeps_accents() -> None:
    assert normalize("A APÓLICE") == "a apólice"


@pytest.mark.unit
def test_tokenize_splits_on_non_word_and_keeps_digits() -> None:
    assert tokenize("r$ 285,00 - cláusula 38n;") == [
        "r",
        "285",
        "00",
        "cláusula",
        "38n",
    ]


@pytest.mark.unit
def test_fold_accents_strips_diacritics() -> None:
    assert fold_accents("indenização") == "indenizacao"
    assert fold_accents("apólice") == "apolice"


@pytest.mark.unit
def test_normalized_tokens_folds_but_does_not_stem() -> None:
    assert normalized_tokens("Perda Total") == ["perda", "total"]


@pytest.mark.unit
def test_analyze_stems_the_accented_token_then_folds() -> None:
    # The order the DoD's stemming depends on: snowball's Portuguese rules are
    # defined over accented text, so the stemmer must see "informação", never
    # the pre-folded "informacao".
    analyzer, stemmer = _analyzer(protected=set())
    analyzer.analyze("Informação")
    assert stemmer.seen == ["informação"]


@pytest.mark.unit
def test_analyze_keeps_a_protected_token_verbatim_and_never_stems_it() -> None:
    analyzer, stemmer = _analyzer(protected={"casco"})
    assert analyzer.analyze("Cobertura CASCO") == ["cobertur", "casco"]
    assert "casco" not in stemmer.seen


@pytest.mark.unit
def test_analyze_preserves_order_and_term_frequency() -> None:
    analyzer, _ = _analyzer(protected=set())
    assert analyzer.analyze("total total total") == ["tota", "tota", "tota"]


@pytest.mark.unit
def test_build_analyzer_uses_snowball_portuguese_stem_then_fold() -> None:
    analyzer = build_analyzer()
    # Accented-stem-then-fold: "indenização" -> "indeniz" (the useful stem);
    # folding first would yield the useless "indenizaca".
    assert analyzer.analyze("a indenização integral") == ["a", "indeniz", "integral"]


@pytest.mark.unit
def test_build_analyzer_query_and_index_sides_agree_on_accented_input() -> None:
    analyzer = build_analyzer()
    assert analyzer.analyze("apólice de carência") == analyzer.analyze(
        "APOLICE DE CARENCIA"
    )


@pytest.mark.unit
def test_build_analyzer_does_not_drop_negation_words() -> None:
    # No stopword removal: "não" must survive -- it flips coverage into exclusion.
    assert "nao" in build_analyzer().analyze("bens não compreendidos no seguro")
