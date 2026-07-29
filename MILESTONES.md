# Milestones

Numeric exit criteria for every milestone. The point of writing
these down is to close a phase on evidence, not on the feeling that it looks
good enough.

Every criterion below is one of three things: a number to measure, a file
that must exist, or a command that must exit zero.

A milestone may close with a **bad** number. That is a valid outcome, not a
failure of process, provided the number was actually measured and the cause
is written down next to it. An unmeasured "looks fine" is worse than a
measured 40%. Undocumented silence about a bad number is the only truly
unacceptable outcome.

## Status

| Milestone | Name                       | Status      |
| --------- | -------------------------- | ----------- |
| M0        | Foundations                | in progress |
| M1        | Policy parsing              | todo        |
| M2        | Ground truth                | todo        |
| M3        | Retrieval                   | todo        |
| M4        | Agent graph                 | todo        |
| M5        | Service and hardening       | todo        |
| M6        | Release and portfolio       | todo        |

---

## M0 — Foundations

Turn a data-only repository into a working project: Python skeleton with
Clean Architecture layers, quality toolchain, CI, and the corpus-integrity
script the documentation already promises. Also fixes the project's scope
statement in writing, in every place a reader or a future prompt might look.

**Exit criteria:**
- `make check` (lint, format, type-check, tests) passes locally and in CI.
- The text-layer audit covers 30/30 documents and its result is recorded in
  the manifest and in `docs/DATA_SOURCES.md`.
- `MILESTONES.md` exists with numeric exit criteria for every milestone.
- The scope statement appears in `README.md`, `NOTICE.md`, `data/README.md`
  and the prompt template directory.

## M1 — Policy parsing

Turn 30 PDFs and 1,709 pages into a clause tree with reliable type metadata.
Retrieval quality is bounded above by parsing quality, so parsing accuracy is
measured and published as a first-class project metric, with its failure
modes described rather than hidden.

**Exit criteria:**
- 30/30 documents produce a clause tree with no unhandled exceptions.
- The two documents without a usable text layer are recovered, with ≥95% of
  their pages yielding non-empty text.
- Stratified manual validation of 50 clauses: ≥90% correct clause
  boundaries, ≥85% correct clause type.
- ≥95% of clauses carry complete provenance (document id, SUSEP process,
  insurer CNPJ, product line, page range).
- `docs/PARSING.md` published with method, per-document results and known
  failures.

## M2 — Ground truth

Build the golden set and the synthetic claim set before any retrieval code
is written, so the golden set does not end up describing what the retriever
already does. Also writes the evaluation protocol, including an honest
account of the fact that one person authors the questions, labels the
answers and builds the system.

**Exit criteria:**
- ≥90 golden questions spanning ≥20 documents, each with reference clause
  ids.
- ≥30 synthetic claims labelled across three verdicts: compatible,
  incompatible, insufficient information — including ≥8 product/claim
  mismatch cases.
- ≥15 questions whose answer is absent from the corpus by construction
  (deductibles, insured amounts, policy periods).
- Blind re-labelling of a 20% sample ≥14 days after first pass; agreement
  rate measured and published in `docs/EVALUATION.md`, whatever it is,
  reference value ≥90% — if below, the cause is documented; disagreements
  resolved and recorded regardless of the rate.
- `docs/EVALUATION.md` complete; golden set tagged `golden-set-v1`.

## M3 — Retrieval

Build and measure the classical RAG pipeline: clause-aware chunking,
embeddings in pgvector, lexical retrieval, hybrid fusion, cross-encoder
reranking and an explicit insufficient-context gate. The hard case this
milestone must handle is the exclusion clause sitting three paragraphs away
from the coverage clause it cancels.

**Exit criteria:**
- Recall@10 and MRR on `golden-set-v1` measured and published for the best
  configuration, reference values Recall@10 ≥ 0.85 and MRR ≥ 0.60 — if
  below, the cause is documented.
- Benchmark matrix committed with all four configurations measured: lexical
  only, dense only, hybrid, hybrid + rerank.
- Exclusion-clause recall reported as a separate number from overall recall.
- Insufficient-context gate precision on the unanswerable subset measured
  and published, reference value ≥80% — if below, the cause is documented.
- Insufficient-context gate recall on the unanswerable subset: 100%,
  enforced by an automated test asserting that every question in the
  unanswerable set triggers the gate, not by review.
- `make build-index` rebuilds the full index from `data/policies/raw/` on a
  clean machine.

## M4 — Agent graph

Assemble the LangGraph state graph: intake with a conditional clarification
loop, retrieval, two assessment nodes running in parallel, recommendation
synthesis, and a persisted human checkpoint before anything is recorded as
final. Every assertion cites a clause id, and deterministic checks run as
Python code rather than at an LLM's discretion.

**Exit criteria:**
- The graph completes on 100% of the synthetic claim set with no unhandled
  exceptions.
- End-to-end verdict accuracy against the three-class labels measured and
  published, reference value ≥75% — if below, the cause is documented with
  a failure catalogue (retrieval miss, parsing error, reasoning error).
- Citation coverage 100%: zero assessment assertions without a clause id,
  enforced by an automated check, not by review.
- Checkpoint verified across a process restart: state persists, resumes,
  and the human decision is recorded alongside the original system
  recommendation.
- The deterministic/LLM boundary in the consistency node is documented in
  `docs/ARCHITECTURE.md`.

## M5 — Service and hardening

The notebook-to-service crossing. Wrap the graph in a FastAPI service
following the Clean Architecture already published in
`fastapi-clean-architecture-api`, with real persistence, asynchronous
processing for long documents, tracing, and a Docker Compose stack anyone
can run. Latency and cost stop being unknowns and become measured numbers.

**Exit criteria:**
- `docker compose up -d` starts api + postgres/pgvector + redis (+ tracing)
  with healthchecks, and the assessment flow works end to end through the
  stack.
- Three endpoints working: submit assessment, read assessment state, submit
  human decision to resume the graph.
- Domain and application layers import no FastAPI, SQLAlchemy, LangGraph or
  LangChain symbols — enforced by an import test, not by convention.
- CI runs lint, type-check, unit tests, integration tests against Postgres,
  and builds the image.
- p95 latency and mean cost per assessment measured and recorded.

## M6 — Release and portfolio

Ship it. Documentation that lets a reviewer understand the system in five
minutes, a demo video of the local UI, a slide deck for LinkedIn, and a
tagged release whose README leads with measured numbers rather than
architecture diagrams.

**Exit criteria:**
- README leads with the results table: parsing accuracy, Recall@10, MRR,
  end-to-end verdict accuracy, latency, cost.
- `docs/` complete: ARCHITECTURE, DATA_SOURCES, PARSING, EVALUATION,
  CONTRACT.
- Demo video recorded and linked; slide deck exported to PDF.
- `v0.1.0` tagged with release notes; CI green on `main`.
