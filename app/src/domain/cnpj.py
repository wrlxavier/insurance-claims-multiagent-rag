"""The CNPJ value object [M5-01].

A CNPJ is the 14-digit Brazilian company registration number -- the
reliable identity key for an insurer, since two companies can share a brand
and differ only here (``data/README.md``; HDI Seguros vs HDI Global).

Validated at construction: exactly 14 digits, and the two trailing digits
must satisfy the standard mod-11 check-digit algorithm. :meth:`Cnpj.parse`
additionally applies the **14-digit zero-padding rule**: SUSEP's open
product catalogue publishes CNPJ as a number rather than a fixed-width
string, so a value that has lost its leading zero and arrives with 13
digits is left-padded before validation (``data/README.md``, "Known
upstream defect: CNPJ leading zeros").

Standard library only -- enforced by
tests/architecture/test_layer_boundaries.py.
"""

from dataclasses import dataclass

from domain.errors import InvalidCnpjError

_PUNCTUATION = str.maketrans("", "", "./- ")
_FIRST_WEIGHTS = (5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2)
_SECOND_WEIGHTS = (6, 5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2)


def _mod11_digit(digits: str, weights: tuple[int, ...]) -> str:
    """One CNPJ check digit: weighted sum, mod 11, ``0`` when the remainder is <2."""
    total = sum(int(d) * w for d, w in zip(digits, weights, strict=True))
    remainder = total % 11
    return "0" if remainder < 2 else str(11 - remainder)


def _cnpj_check_digits(base12: str) -> str:
    """The two check digits for a 12-digit CNPJ base, as a 2-character string."""
    first = _mod11_digit(base12, _FIRST_WEIGHTS)
    second = _mod11_digit(base12 + first, _SECOND_WEIGHTS)
    return first + second


@dataclass(frozen=True)
class Cnpj:
    """A Brazilian company registration number: exactly 14 ASCII digits.

    Construct from an already-canonical 14-digit string, or use
    :meth:`parse` for any other form (punctuated, or the 13-digit
    leading-zero-stripped form the upstream catalogue emits).
    """

    value: str

    def __post_init__(self) -> None:
        """Reject a non-14-digit value or a value whose check digits do not verify."""
        if len(self.value) != 14 or not (self.value.isascii() and self.value.isdigit()):
            raise InvalidCnpjError(f"not a 14-digit CNPJ: {self.value!r}")
        expected = _cnpj_check_digits(self.value[:12])
        if self.value[12:] != expected:
            raise InvalidCnpjError(
                f"CNPJ check digits do not verify: {self.value!r} "
                f"(expected ...{expected})"
            )

    @classmethod
    def parse(cls, raw: str) -> "Cnpj":
        """Normalise ``raw`` to 14 canonical digits, then validate.

        Strips ``.``, ``/``, ``-`` and whitespace; left-pads a 13-digit
        result with one zero (the upstream catalogue defect). A stripped
        length other than 13 or 14 raises
        :class:`domain.errors.InvalidCnpjError`.
        """
        digits = raw.translate(_PUNCTUATION)
        if len(digits) == 13:
            digits = "0" + digits
        if len(digits) != 14:
            raise InvalidCnpjError(
                f"cannot normalise to 14 digits: {raw!r} -> {digits!r}"
            )
        return cls(digits)

    @property
    def formatted(self) -> str:
        """The punctuated form ``NN.NNN.NNN/NNNN-NN``."""
        v = self.value
        return f"{v[:2]}.{v[2:5]}.{v[5:8]}/{v[8:12]}-{v[12:]}"
