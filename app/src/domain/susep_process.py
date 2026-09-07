"""The SUSEP process number value object [M5-01].

A SUSEP process number identifies a *registered product* -- not a single
document and not an insurance contract (``data/README.md``, "Version
pinning"). Its canonical written form is ``NNNNN.NNNNNN/NNNN-NN``, e.g.
``15414.610650/2024-59``; the corpus stores the same 17 digits with the
punctuation stripped as each document's filename stem.

Format is validated at construction. The trailing ``-NN`` is *not*
check-digit verified: SUSEP's process check-digit algorithm is not a
stable, published specification (unlike the CNPJ mod-11 in
[domain.cnpj]), and a false rejection of a real filing is unrecoverable.
The one regex in the codebase that touched this (``scripts/eval_intake.py``)
is likewise format-only.

Standard library only -- enforced by
tests/architecture/test_layer_boundaries.py.
"""

import re
from dataclasses import dataclass

from domain.errors import InvalidSusepProcessError

_CANONICAL = re.compile(r"\d{5}\.\d{6}/\d{4}-\d{2}")
_SEVENTEEN_DIGITS = re.compile(r"\d{17}")


@dataclass(frozen=True)
class SusepProcess:
    """A SUSEP process number in canonical ``NNNNN.NNNNNN/NNNN-NN`` form.

    Construct from an already-canonical string, or use :meth:`parse` for any
    other form (the 17-digit filename stem, or a value with surrounding
    whitespace).
    """

    value: str

    def __post_init__(self) -> None:
        """Reject anything but the exact canonical format."""
        if not _CANONICAL.fullmatch(self.value):
            raise InvalidSusepProcessError(
                f"not a SUSEP process number (expected NNNNN.NNNNNN/NNNN-NN): "
                f"{self.value!r}"
            )

    @classmethod
    def parse(cls, raw: str) -> "SusepProcess":
        """Normalise ``raw`` to canonical form, then validate.

        Accepts the canonical form unchanged, or 17 bare digits (the
        manifest filename stem); trims surrounding whitespace first. Anything
        else raises :class:`domain.errors.InvalidSusepProcessError`.
        """
        stripped = raw.strip()
        if _SEVENTEEN_DIGITS.fullmatch(stripped):
            stripped = (
                f"{stripped[:5]}.{stripped[5:11]}/{stripped[11:15]}-{stripped[15:]}"
            )
        return cls(stripped)

    @property
    def digits(self) -> str:
        """The 17 digits with no punctuation, e.g. ``"15414610650202459"``."""
        return self.value.translate(str.maketrans("", "", "./-"))

    @property
    def filename_stem(self) -> str:
        """The document filename without its ``.pdf`` suffix -- the same 17 digits."""
        return self.digits

    @property
    def year(self) -> str:
        """The four-digit year in the ``/NNNN`` group, e.g. ``"2024"``."""
        return self.value[13:17]
