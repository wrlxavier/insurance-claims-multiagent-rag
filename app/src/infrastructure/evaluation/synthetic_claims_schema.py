"""The synthetic claim set schema [M2-04].

Fixes the format every synthetic claim narrative must follow, mirroring
[infrastructure.evaluation.golden_set_schema]'s ``GoldenQuestion`` pattern: a
flat, validated Pydantic row that fails loudly on a malformed claim rather
than accepting it silently.

Authorship is the same three-layer flow ``GoldenQuestion`` documents: a
deterministic Layer 1 (``scripts/synthetic_claims_selection.py``) picks the
target document, source clause(s), and verdict; an LLM (Layer 2,
``scripts/draft_synthetic_claims.py``) phrases the narrative; the author
verifies both in ``eval/synthetic_claims_draft.csv`` before a row is
promoted here by ``scripts/finalize_synthetic_claims_from_review.py``.

Unlike a golden question, a synthetic claim never asks about the corpus --
it is free text a policyholder would write, which the (future) intake node
has to make sense of. ``missing_fact_type`` records, for
``insufficient_information`` claims only, which fact the narrative
deliberately omits -- the ground truth the M4-03 clarification loop's
``missing_information`` output should eventually reproduce.
"""

from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field, model_validator

from infrastructure.evaluation.golden_set_schema import ExpectedVerdict

SCHEMA_VERSION = "v1"

_NonEmptyStr = Annotated[str, Field(min_length=1)]


class MissingFactType(Enum):
    """The load-bearing fact an insufficient_information claim omits.

    Mirrors [scripts.synthetic_claims_selection.MISSING_FACT_PATTERNS]'s keys
    verbatim.
    """

    AMBITO_GEOGRAFICO = "ambito_geografico"
    USO_DO_VEICULO = "uso_do_veiculo"
    DATA_EVENTO_VIGENCIA = "data_evento_vigencia"
    VALOR_FRANQUIA_LIMITE = "valor_franquia_limite"
    TIPO_EVENTO_CONDICAO = "tipo_evento_condicao"


class SyntheticClaim(BaseModel):
    """One synthetic claim-narrative row.

    ``expected_verdict`` reuses [golden_set_schema.ExpectedVerdict] verbatim
    -- the project's one verdict vocabulary applies here exactly as it does
    to a golden question. ``missing_fact_type`` is required exactly when
    ``expected_verdict`` is ``insufficient_information`` and forbidden
    otherwise -- the claim narrative's own missing fact, not a property of
    the corpus.
    """

    schema_version: str
    claim_id: _NonEmptyStr
    document_id: _NonEmptyStr
    narrative: _NonEmptyStr
    reference_clause_ids: list[str]
    expected_verdict: ExpectedVerdict
    missing_fact_type: MissingFactType | None = None
    notes: str = ""
    authored_at: str | None = None

    @model_validator(mode="after")
    def _check_missing_fact_consistency(self) -> "SyntheticClaim":
        if self.expected_verdict == ExpectedVerdict.INSUFFICIENT_INFORMATION:
            if self.missing_fact_type is None:
                raise ValueError(
                    "insufficient_information claims must set missing_fact_type"
                )
        elif self.missing_fact_type is not None:
            raise ValueError(
                f"{self.expected_verdict.value} claims must not set "
                f"missing_fact_type, got {self.missing_fact_type!r}"
            )
        if not self.reference_clause_ids:
            raise ValueError("claims must have at least one reference_clause_ids entry")
        return self
