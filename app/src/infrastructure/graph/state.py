"""The agent graph's state schema and its fan-in reducer [M4-01].

Modelled once, here, so traceability is structural rather than bolted on:
every M4 node reads and writes this ``ClaimState``, and every node appends to
``audit_trail``. Keeping the shape in one module means M4-02..M4-09 extend a
known contract instead of renegotiating it.

Layer note: this lives in ``infrastructure`` because it uses Pydantic (and,
downstream, LangGraph) -- both forbidden in ``domain``/``application`` by
tests/architecture/test_layer_boundaries.py. The domain-layer entities this
mirrors (``Claim``, ``Assessment``, ``HumanDecision``) arrive in M5-01 as
stdlib dataclasses, with mappers between the two; the one piece already
shared is ``domain.verdict.Verdict``.

This module imports **no** langgraph: it is the pure data contract. Graph
assembly -- nodes, edges, the fixed parallel branches ([M4-07], *not*
``Send``), the checkpointer ([M4-09]) -- is what imports langgraph, from
M4-02 onward.

Supersedes the draft in
``.ai_context/Assistente_Sinistros_Apolices_Proposta_Completa_com_ERRATA.md``
sec. 6.3 on the two points its own ERRATA flags: verdicts are the M0-06
vocabulary (``domain.verdict.Verdict``), not ``coberto``/``excluido``; and
the parallel fan-in needs a channel reducer, not ``Send``.
"""

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Annotated, Literal, NamedTuple, TypedDict

from pydantic import BaseModel, ConfigDict, Field, model_validator

from domain.clause_classification import ClauseType
from domain.verdict import Verdict

SCHEMA_VERSION = "v1"

_NonEmptyStr = Annotated[str, Field(min_length=1)]
_Confidence = Annotated[float, Field(ge=0.0, le=1.0)]


class Citation(BaseModel):
    """One clause an assertion is traceable to.

    ``excerpt`` is a quoted span of the clause as a human reads it -- the
    chunk ``display_text`` ([infrastructure.rag.chunk_schema]), never the
    breadcrumb-prefixed string the embedding model saw. ``relevance_score``
    is the ranker's score for this clause against the query; the retrieval
    node ([M4-04]) supplies it (the retrieval interface itself returns only
    clause ids).
    """

    model_config = ConfigDict(frozen=True)

    clause_id: _NonEmptyStr
    document_id: _NonEmptyStr
    susep_process: _NonEmptyStr
    clause_type: ClauseType
    relevance_score: float = Field(ge=0.0)
    excerpt: str


class TokenUsage(BaseModel):
    """Token counts for one LLM call, as LangChain's ``usage_metadata`` reports them."""

    model_config = ConfigDict(frozen=True)

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)


class AuditEvent(BaseModel):
    """One entry in the append-only audit trail.

    A deterministic node ([M4-06]'s Python checks) leaves ``model``,
    ``model_version``, ``token_usage`` and ``confidence`` as ``None`` -- the
    absence is itself the record that no model was consulted. ``node_input``
    is a compact string the node chooses to record (the query it built, the
    fields it read), not a dump of the whole state.
    """

    model_config = ConfigDict(frozen=True)

    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    node: _NonEmptyStr
    action: _NonEmptyStr
    model: str | None = None
    model_version: str | None = None
    token_usage: TokenUsage | None = None
    confidence: _Confidence | None = None
    node_input: str | None = None


class AuditRecord(NamedTuple):
    """One audit event on its way to durable storage, with optional detail.

    The pair [infrastructure.graph.context.AuditTrailSink] persists. ``payload``
    carries structure the flat ``AuditEvent`` has no field for -- [M4-09] uses it
    for the analyst's whole ``HumanDecision`` (notes, ``decided_at``, any
    ``edited_recommendation``), so "what the human decided" is answerable in SQL
    without going through LangGraph's checkpoint serde. ``None`` for every event
    a node produced on its own.

    A plain ``NamedTuple``, not a Pydantic model: it never enters graph state and
    never crosses the serializer -- it exists only between the checkpoint node
    and the sink.
    """

    event: AuditEvent
    payload: dict[str, object] | None = None


class ExtractedEntities(BaseModel):
    """Structured entities the intake node ([M4-02]) pulls from the free-text claim.

    Every field is optional: intake is explicit about what is missing rather
    than inventing a value, and the gaps drive the clarification loop
    ([M4-03]). ``product_line`` is intake's classification of the event
    against the corpus's product lines, not a value copied from the text.
    """

    model_config = ConfigDict(frozen=True)

    event_type: str | None = None
    event_date: str | None = None
    description: str | None = None
    estimated_amount: float | None = None
    vehicle_info: str | None = None
    susep_process: str | None = None
    product_line: str | None = None


class ClarificationQuestion(BaseModel):
    """One question the clarification loop ([M4-03]) put to the claimant.

    ``field`` is one of ``schemas.MissingInfoTag``'s values -- typed ``str``
    here, not that ``Literal``, so ``state.py`` keeps importing nothing from
    ``schemas.py`` (the two schemas are deliberately separate). The value is
    constrained to the tag set upstream, by ``schemas.ClarificationOutput``.
    The clarification node accumulates these across rounds; there is no channel
    reducer because only that one sequential node ever writes them.
    """

    model_config = ConfigDict(frozen=True)

    field: _NonEmptyStr
    question: _NonEmptyStr


class CompatibilityAssessment(BaseModel):
    """The compatibility node's ([M4-05]) structured verdict.

    Every assertion in ``reasoning`` must be traceable to a clause in
    ``citations``; a citation-free assertion is a malformed output the node
    rejects and retries, not something patched here.
    """

    model_config = ConfigDict(frozen=True)

    verdict: Verdict
    reasoning: str
    citations: list[Citation]
    confidence: _Confidence


class ConsistencySignal(BaseModel):
    """One thing the consistency node ([M4-06]) flags for human attention.

    Never a verdict -- this node signals, it does not decide. ``source``
    records whether a deterministic Python check or the LLM raised it, so the
    two stay measurable apart (the boundary is documented in
    ``docs/ARCHITECTURE.md`` by [M4-06]).
    """

    model_config = ConfigDict(frozen=True)

    check: _NonEmptyStr
    severity: Literal["info", "attention"]
    detail: str
    source: Literal["deterministic", "llm"]


class ConsistencyReport(BaseModel):
    """The consistency node's ([M4-06]) full output: a set of signals, no verdict."""

    model_config = ConfigDict(frozen=True)

    signals: list[ConsistencySignal]


class Recommendation(BaseModel):
    """The recommendation node's ([M4-08]) consolidated opinion for a reviewer.

    ``citations`` may only contain clauses an upstream node already produced
    ([M4-08] verifies this by test). ``consistency_flags`` stay separate from
    the compatibility verdict -- attention points, not part of the decision.
    """

    model_config = ConfigDict(frozen=True)

    recommended_action: str
    justification: str
    citations: list[Citation]
    consistency_flags: list[ConsistencySignal]
    confidence: _Confidence


class HumanDecision(BaseModel):
    """The analyst's decision at the checkpoint ([M4-09]).

    Recorded alongside -- never overwriting -- the system's original
    ``recommendation``. ``edited_recommendation`` carries the analyst's
    revised opinion and is required exactly when ``decision`` is ``"edit"``.
    """

    model_config = ConfigDict(frozen=True)

    decision: Literal["approve", "edit", "reject"]
    notes: str = ""
    decided_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    edited_recommendation: Recommendation | None = None

    @model_validator(mode="after")
    def _check_edit_carries_a_revision(self) -> "HumanDecision":
        if self.decision == "edit" and self.edited_recommendation is None:
            raise ValueError("decision 'edit' requires edited_recommendation")
        if self.decision != "edit" and self.edited_recommendation is not None:
            raise ValueError(
                f"decision {self.decision!r} must not carry edited_recommendation"
            )
        return self


def append_audit_events(
    left: Sequence[AuditEvent], right: Sequence[AuditEvent]
) -> list[AuditEvent]:
    """Fan-in reducer for ``audit_trail``: concatenate, accumulated first.

    ``audit_trail`` is the one channel written by every node and -- in
    [M4-07]'s topology -- by the two assessment nodes in the same superstep.
    Without a reducer LangGraph raises ``InvalidUpdateError`` on that
    concurrent write; with this one both branches' events survive, in a
    deterministic order (the existing trail, then this superstep's
    additions). No dedup: a repeated event is a real signal (e.g. [M4-09]
    re-running a node after an interrupt), not noise to hide.
    """
    return [*left, *right]


class _ClaimStateIn(TypedDict):
    """The fields always supplied when the graph is invoked."""

    claim_id: str
    raw_claim_text: str


class ClaimState(_ClaimStateIn, total=False):
    """The agent graph's shared state.

    ``audit_trail`` is the only channel with a reducer: the parallel
    assessment nodes write disjoint fields otherwise (``compatibility`` vs
    ``consistency``), and ``entities`` / ``missing_information`` are rewritten
    by the sequential clarification passes, where last-write-wins is intended.
    [M4-03]'s ``clarification_questions`` is accumulated by the clarification
    node itself (``prior + new``), which is the "own accumulator" this note
    anticipated -- still no channel reducer, because that one sequential node
    is its only writer.
    """

    # intake + clarification loop -- M4-02, M4-03
    entities: ExtractedEntities | None
    missing_information: list[str]
    clarification_rounds: int
    # every question the loop has asked, across every round -- accumulated by
    # the clarification node, no reducer (sole writer, runs sequentially)
    clarification_questions: list[ClarificationQuestion]
    # set once by the terminal clarification_exhausted node when the loop hits
    # MAX_CLARIFICATION_ROUNDS with gaps still open: "terminate as
    # insufficient_information, gaps listed" (the gaps stay in
    # missing_information). A plain bool, not a Literal outcome -- there is no
    # "resolved" writer or consumer to justify one yet.
    clarification_exhausted: bool
    # retrieval -- M4-04
    citations: list[Citation]
    context_sufficient: bool | None
    # parallel assessment -- M4-05, M4-06 (wired as fixed branches by M4-07)
    compatibility: CompatibilityAssessment | None
    consistency: ConsistencyReport | None
    # synthesis + human checkpoint -- M4-08, M4-09
    recommendation: Recommendation | None
    human_decision: HumanDecision | None
    # written by every node, incl. both parallel branches -- carries a reducer
    audit_trail: Annotated[list[AuditEvent], append_audit_events]
