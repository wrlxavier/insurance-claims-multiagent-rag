"""Loader for the committed stemming-exception list -- [M3-03]."""

from pathlib import Path

import pytest

from infrastructure.rag.lexical_config import LEXICAL_STEMMING_EXCEPTIONS_PATH
from infrastructure.rag.lexical_stemming_exceptions import load_stemming_exceptions

_REPO_ROOT = Path(__file__).resolve().parents[4]


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "exceptions.csv"
    path.write_text(body, encoding="utf-8")
    return path


@pytest.mark.unit
def test_loads_terms_normalised_and_ignores_the_note_column(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        'term,note\nCASCO,a code\ncarência,"has, a comma in the note"\n',
    )
    assert load_stemming_exceptions(path) == frozenset({"casco", "carencia"})


@pytest.mark.unit
def test_a_multi_token_term_is_a_data_error(tmp_path: Path) -> None:
    path = _write(tmp_path, 'term,note\nperda total,"BM25 is bag-of-words"\n')
    with pytest.raises(ValueError, match="exactly one token"):
        load_stemming_exceptions(path)


@pytest.mark.unit
def test_missing_file_fails_loudly(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_stemming_exceptions(tmp_path / "nope.csv")


@pytest.mark.unit
def test_returns_a_frozenset() -> None:
    assert isinstance(
        load_stemming_exceptions(_REPO_ROOT / LEXICAL_STEMMING_EXCEPTIONS_PATH),
        frozenset,
    )


@pytest.mark.unit
def test_the_committed_list_is_single_tokens_and_carries_the_negation_fix() -> None:
    protected = load_stemming_exceptions(_REPO_ROOT / LEXICAL_STEMMING_EXCEPTIONS_PATH)
    assert protected  # non-empty
    assert all(term.isascii() and " " not in term for term in protected)
    # The one measurement-driven entry: 'não' -> 'na' collision fix.
    assert "nao" in protected
