"""The policy entity: a registered insurance product [M5-01].

"Policy" here is SUSEP's *condições gerais* -- the registered product a
claim is assessed against -- not an insurance contract and not a single
document (a product can have several filed versions; ``data/README.md``,
"Version pinning"). It promotes the seven bare-string fields of
``domain.clause_classification.ClauseProvenance`` into an entity, with the
two identifiers as validated value objects.

``ClauseProvenance`` is left untouched: the parsing pipeline is frozen and
``Policy`` is additive. ``product_line`` and ``indemnity_regime`` stay
plain strings -- ``data/README.md`` keeps the Portuguese SUSEP codes
verbatim so the corpus still joins back to the catalogue.

Standard library only -- enforced by
tests/architecture/test_layer_boundaries.py.
"""

from dataclasses import dataclass

from domain.cnpj import Cnpj
from domain.errors import PolicyYearMismatchError
from domain.susep_process import SusepProcess


@dataclass(frozen=True)
class Policy:
    """A registered motor-insurance product, keyed by its SUSEP process."""

    susep_process: SusepProcess
    cnpj: Cnpj
    insurer: str
    product_line: str
    indemnity_regime: str
    process_year: str

    def __post_init__(self) -> None:
        """Reject empty descriptive fields, or a year the process disagrees with."""
        for name in ("insurer", "product_line", "indemnity_regime", "process_year"):
            if not getattr(self, name):
                raise ValueError(f"Policy.{name} must not be empty")
        if self.process_year != self.susep_process.year:
            raise PolicyYearMismatchError(
                f"process_year {self.process_year!r} does not match "
                f"{self.susep_process.value!r} (year {self.susep_process.year!r})"
            )

    @classmethod
    def from_manifest_row(cls, row: dict[str, str]) -> "Policy":
        """Build from a ``data/policies/manifest.csv`` record.

        ``Cnpj.parse`` applies the 14-digit zero-padding rule and
        ``SusepProcess.parse`` accepts either written form. Mirrors
        ``infrastructure.rag.retrieval_filter.RetrievalFilter.from_manifest_row``.
        """
        return cls(
            susep_process=SusepProcess.parse(row["susep_process"]),
            cnpj=Cnpj.parse(row["cnpj"]),
            insurer=row["insurer"],
            product_line=row["product_line"],
            indemnity_regime=row["indemnity_regime"],
            process_year=row["process_year"],
        )

    @property
    def identity(self) -> SusepProcess:
        """The semantic key: the SUSEP process identifies the product.

        ``__eq__`` still compares every field; this names the key the
        repositories ([M5-02]) look a policy up by.
        """
        return self.susep_process
