# Domain entities

The [M5-01] business layer: the nouns the service is about and the rules they
must always satisfy, modelled with **no framework in sight**. Everything here is
in `app/src/domain/`, imports nothing but the standard library and `typing`, and
is a `@dataclass(frozen=True)` or an `enum.Enum`. `tests/architecture/
test_layer_boundaries.py` fails CI if a domain module ever imports Pydantic,
SQLAlchemy, LangGraph or LangChain.

**Why it exists as its own layer.** M5 wraps the agent graph in a service
([M5-02] ports, [M5-03] persistence, [M5-04] API). The entities here are the
currency those layers are written in — a repository returns an `Assessment`, a
use case takes a `Claim`, the API serialises a `HumanDecision`. Keeping them
framework-free means the business rules are testable by construction, in
microseconds, with no graph, no database and no HTTP.

**Relationship to the agent graph.** `infrastructure/graph/state.py` holds
Pydantic *twins* of several of these types (`Citation`, `CompatibilityAssessment`,
`HumanDecision`, …) — it has to, because LangGraph needs Pydantic and Pydantic is
forbidden in `domain/`. The graph keeps producing those inside a run; the domain
dataclasses are the persisted, API-facing shape. The aggregate ↔ ORM-row mapper
is [M5-03]'s (`infrastructure/database/assessment_mapper.py`); the graph-state ↔
domain mapper is [M5-04]'s, with the orchestrator adapter that needs it (see
"Deferred"). `domain.verdict.Verdict` is the one type already shared by both,
unchanged since [M4-01].

---

## Value objects

### `SusepProcess` — `domain/susep_process.py`

A SUSEP process number identifies a **registered product**, not a single document
and not an insurance contract (`data/README.md`, "Version pinning"). Canonical
written form `NNNNN.NNNNNN/NNNN-NN`, e.g. `15414.610650/2024-59`.

| Member | Purpose |
| --- | --- |
| `value: str` | The canonical string. `__post_init__` rejects anything that is not an exact `re.fullmatch` of `\d{5}\.\d{6}/\d{4}-\d{2}` → `InvalidSusepProcessError`. |
| `parse(raw) -> SusepProcess` | Normalises then validates: trims whitespace, and expands the 17-digit filename stem (`15414610650202459`) back to canonical form. |
| `.digits` / `.filename_stem` | The 17 digits with no punctuation — equals the corpus filename minus `.pdf`. |
| `.year` | The `/NNNN` group, e.g. `"2024"` — equals the manifest `process_year` column. |

The trailing `-NN` is **not** check-digit verified. SUSEP's process check-digit
algorithm is not a published, stable specification (unlike CNPJ's), and a false
rejection of a real filing is unrecoverable. The only pre-existing SUSEP regex in
the repo (`scripts/eval_intake.py`) is likewise format-only.

### `Cnpj` — `domain/cnpj.py`

The 14-digit Brazilian company registration number — the reliable identity key
for an insurer, since two companies can share a brand and differ only here
(`data/README.md`; HDI Seguros vs HDI Global).

| Member | Purpose |
| --- | --- |
| `value: str` | Exactly 14 ASCII digits. `__post_init__` checks the length, that every character is a digit, **and** that the two trailing digits satisfy the standard mod-11 check-digit algorithm → `InvalidCnpjError`. |
| `parse(raw) -> Cnpj` | Strips `.` `/` `-` and whitespace, then applies the **14-digit zero-padding rule** — left-pads a 13-digit value with one `0` — then validates. A stripped length other than 13 or 14 raises. |
| `.formatted` | The punctuated form `NN.NNN.NNN/NNNN-NN`. |
| `_cnpj_check_digits(base12)` | Module-private; the mod-11 helper, exposed so tests can synthesise valid values. |

**The zero-padding rule.** SUSEP's open product catalogue publishes CNPJ as a
number rather than a fixed-width string, so 41 of its 180 motor-line rows have
lost a leading zero and carry 13 digits (`data/README.md`, "Known upstream
defect"). `data/policies/manifest.csv` is already corrected; `Cnpj.parse` is what
makes any *future* read from that catalogue correct too. All 30 corpus CNPJs pass
the checksum — a parametrized test in `test_cnpj.py` asserts it.

The checksum trade-off: a hand-rolled random 14-digit fixture will be rejected.
That is the value object doing its job — fixtures use real corpus CNPJs or
`_cnpj_check_digits` to build valid ones.

---

## Entities

### `Policy` — `domain/policy.py`

A registered motor-insurance product (SUSEP *condições gerais*). Promotes the
seven bare-string fields of `domain.clause_classification.ClauseProvenance` into
an entity with validated identifiers. `ClauseProvenance` is left untouched — the
parsing pipeline is frozen and `Policy` is additive.

| Field | Type | Notes |
| --- | --- | --- |
| `susep_process` | `SusepProcess` | The identity (`.identity` returns it). "Identifies the registered product, not a single document." |
| `cnpj` | `Cnpj` | The filing company. |
| `insurer` | `str` | Non-empty; verbatim legal name. |
| `product_line` | `str` | Non-empty; `CASCO` / `RCF-A` / `ASSIST` / `GAR.EST` / `CARTA VERDE`. Kept a string, not an enum — `data/README.md` keeps the Portuguese SUSEP codes verbatim so the corpus joins back to the catalogue. |
| `indemnity_regime` | `str` | Non-empty; `VD` / `VMR` / `VD+VMR` / `n/a`. |
| `process_year` | `str` | **Invariant:** must equal `susep_process.year` → `PolicyYearMismatchError`. |

`Policy.from_manifest_row(row)` builds one from a `data/policies/manifest.csv`
record (mirrors `RetrievalFilter.from_manifest_row`); `Cnpj.parse` applies the
zero-pad there.

### `PolicyClause` — `domain/policy_clause.py`

One clause of a registered product, as the service layer sees it: a stable id,
its business type, its text. **Distinct from `domain.clause_tree.Clause`** — that
is an 18-field parse-tree node (parent/child pointers, page spans, per-line page
attribution), a build-time artifact cached to Parquet and never persisted.
[M5-02]'s `ClauseRepository` returns `PolicyClause`; [M5-03]'s
`SqlAlchemyClauseRepository` projects one from the `chunk` table (grouping the
rows whose `source_clause_ids` carries the wanted id), not from the clause tree —
the corpus is already indexed there and that is the granularity a `Citation`
refers to.

The DoD names this entity `Clause`; it is `PolicyClause` to avoid shadowing the
existing name (`docs/ARCHITECTURE.md` records the deviation).

| Field | Type | Notes |
| --- | --- | --- |
| `clause_id` | `str` | Non-empty; the `f"{document_id}:{path}"` convention. |
| `susep_process` | `SusepProcess` | The registered product it belongs to. |
| `document_id` | `str` | Non-empty; which parsed filing/version. |
| `clause_type` | `ClauseType` | `isinstance`-guarded. |
| `text` | `str` | Non-empty. |
| `heading` | `str` | Optional (numbering label + title), for the API. |

### `Citation` — `domain/citation.py`

The stdlib twin of `state.Citation`: one clause an assessment's reasoning is
grounded in. `clause_id`, `document_id`, `excerpt` non-empty; `susep_process` a
`SusepProcess`; `clause_type` a `ClauseType`; `relevance_score` a float `>= 0.0`
defaulting to `0.0` (a structurally co-retrieved exclusion the ranker missed
carries `0.0` — `docs/ARCHITECTURE.md`, M3-06).

### `Claim` — `domain/claim.py`

The free-text loss narrative a policyholder submits, before the graph makes
structured sense of it.

| Field | Type | Notes |
| --- | --- | --- |
| `claim_id` | `str` | Non-empty. |
| `raw_text` | `str` | Non-empty; `ClaimState.raw_claim_text`. |
| `submitted_at` | `datetime` | **Must be timezone-aware.** |
| `policy_ref` | `SusepProcess \| None` | The registered product the claim is filed against. `None` is the honest default today — intake *extracts* it from the narrative; [M5-04] makes it a first-class submission field. |

The structured read intake produces (`ExtractedEntities`) stays in
`infrastructure/graph/state.py` — it is graph working state, not a business
entity the DoD names.

### `Assessment` — `domain/assessment.py`

The settled compatibility assessment for one claim — the twin of
`state.CompatibilityAssessment` plus the `recommended_action` the reviewer acts
on (so `GET /v1/assessments/{id}` returns one entity, not two).

| Field | Type | Notes |
| --- | --- | --- |
| `assessment_id` | `str` | Non-empty; caller/DB supplied — the domain does not mint ids. |
| `claim_id` | `str` | Non-empty; reference by id, never the `Claim` object. |
| `verdict` | `Verdict` | **Invariant:** must be a `Verdict` member → `VerdictNotPermittedError`. |
| `reasoning` | `str` | Non-empty; rendered prose. |
| `citations` | `tuple[Citation, ...]` | **Invariant:** length `>= 1`, always → `CitationRequiredError(assessment_id)`. |
| `confidence` | `float` | In `[0.0, 1.0]`. |
| `recommended_action` | `str` | Non-empty. |

Consistency signals are out of scope — no M5-01 invariant touches them.

### `HumanDecision` — `domain/human_decision.py`

What the analyst decided at the checkpoint ([M4-09]), recorded alongside — never
overwriting — the system's `Assessment`. `DecisionOutcome` is the enum
(`APPROVE` / `EDIT` / `REJECT`; values match `state.HumanDecision.decision`'s
`Literal` so [M5-03]'s mapper is a one-liner).

| Field | Type | Notes |
| --- | --- | --- |
| `assessment_id` | `str` | **Invariant:** non-empty — a decision always references the assessment it acted on → `DecisionMustReferenceAssessmentError`. |
| `decision` | `DecisionOutcome` | `isinstance`-guarded. |
| `decided_at` | `datetime` | **Must be timezone-aware.** |
| `notes` | `str` | Optional. |
| `edited_assessment` | `Assessment \| None` | **Invariant:** present iff `decision is EDIT`; when present, its `assessment_id` must equal this decision's. Mirrors `state.py`'s `_check_edit_carries_a_revision`. |

---

## Invariants and their errors

`domain/errors.py` holds the hierarchy. `InvalidValueObjectError` is also a
`ValueError` (so `except ValueError` works on a bad constructor argument the
conventional way); `InvariantViolationError` is a `DomainError` only.

| Invariant | Where | Error |
| --- | --- | --- |
| A verdict is one of the three permitted values | `Assessment` | `VerdictNotPermittedError` |
| An assessment always has ≥1 citation | `Assessment` | `CitationRequiredError` (carries `assessment_id`) |
| A decision always references the assessment it acted on | `HumanDecision` | `DecisionMustReferenceAssessmentError` |
| An edit revises the same assessment it references | `HumanDecision` | `ValueError` |
| A SUSEP process matches the canonical format | `SusepProcess` | `InvalidSusepProcessError` |
| A CNPJ is 14 digits with valid check digits | `Cnpj` | `InvalidCnpjError` |
| A policy's `process_year` agrees with its process number | `Policy` | `PolicyYearMismatchError` |
| Timestamps are timezone-aware | `Claim`, `HumanDecision` | `ValueError` |

Bare `ValueError` (not a domain type) is used for field-level checks — an empty
string, a score below zero, a non-enum passed where an enum belongs — matching
the rest of `domain/`.

---

## Deferred

- **Consistency signals on `Assessment`** — until a use case needs
  `ConsistencyReport` persisted ([M5-03]). [M5-02] carries them as
  `application.consistency_flag.ConsistencyFlag` on the `AssessmentRecord`
  aggregate, not on the domain entity.
- **The abstain outcome** — a verdict with no citations is not a domain
  `Assessment` (the ≥1-citation rule is unconditional). [M5-02] models it as
  `application.assessment_record.AssessmentRecord`, the servable/persistable
  aggregate; `AssessmentRecord.as_domain_assessment()` is the grounded
  projection and raises `CitationRequiredError` on an abstain.
- **Editing an assessment** — an `edit` `HumanDecision` always carries a
  *grounded* `Assessment` ([M5-02]'s `SubmitHumanDecision` builds it from the
  reviewer's payload and validates every cited clause); the domain rule is
  unchanged.
- **`ExtractedEntities`** — stays graph working state in `infrastructure/graph/
  state.py`; not a DoD entity.
- **Mappers** — domain/aggregate ↔ ORM row landed in [M5-03]
  (`app/src/infrastructure/database/assessment_mapper.py`). The domain ↔
  `state.py` Pydantic mapper is deferred to [M5-04]: its only consumer is the
  LangGraph orchestrator adapter, and `AssessmentRecord.from_orchestrator_result`
  already bridges the graph-free DTO — this codebase does not land a mapper
  ahead of its caller (the same call that dropped `RetrievalService` in [M5-02]).
- **`Claim.policy_ref` as a required field** — [M5-04], when the submission API
  takes it explicitly.
