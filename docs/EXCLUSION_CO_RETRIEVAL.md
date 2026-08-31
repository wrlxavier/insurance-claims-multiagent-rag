# Exclusion co-retrieval

The [M3-06] stage: after ranking, for every retrieved `coverage` clause, pull
the `exclusion` clauses structurally linked to it and reserve a slot of the
final context so a higher-scoring coverage passage cannot squeeze the exclusion
that limits it out. The contract lives in code at
`app/src/infrastructure/rag/exclusion_co_retrieval_config.py`; the domain rule is
stated in `docs/ARCHITECTURE.md`; this document is the method, the evidence and
the numbers.

**Why this is the core retrieval problem of the domain.** A coverage clause
retrieved without the exclusion that cancels it produces an assessment that is
fluent, well-cited and wrong. `coverage_with_exclusion` is the golden set's
weakest question type by a wide margin — its reference sets carry *both* the
coverage clause and the limiting exclusion on purpose, and lexical + dense
retrieval reliably return the first and miss the second.

**Scope.** This issue adds the co-retrieval step behind the existing
`retrieve(question, *, k, metadata_filter)` interface, on the [M3-05] hybrid RRF
+ rerank + SUSEP-process/CNPJ base, and tunes its one knob — the **reserved
slot count** — on `golden-set-v1`. It does **not**:

- produce the committed lexical / dense / hybrid / hybrid+rerank(+co-retrieval)
  benchmark matrix or `make build-index` — that is [M3-08]'s;
- re-tune the reranker depth, BM25 `k1`/`b` or the RRF constant — those stay
  their pinned `docs/RERANKING.md` / `docs/LEXICAL_RETRIEVAL.md` /
  `docs/HYBRID_RETRIEVAL.md` values;
- add the insufficient-context gate ([M3-07]);
- improve recall of the **coverage** clause itself — see Limitations.

---

## Method

### The structural clause graph

`ClauseGraph` (`exclusion_co_retrieval.py`) is a pure, deterministic index over
the parsed corpus (`build/parsed_clauses.jsonl`, the same `ParsedClauseRecord`
rows the eval harness already loads) — no database, no model. For a retrieved
`coverage` clause it returns the linked `exclusion` clauses by three edge types,
which are exactly the three the [M3-06] DoD names:

- **`same_section`** — the exclusion shares the coverage clause's top-level
  section root (walk `parent_id` to the root). This subsumes siblings (same
  `parent_id`), descendants of the coverage clause, and exclusions elsewhere
  in the same numbered section. When the coverage clause carries a
  `bundle_section` the candidate must match it (or be `None`). *Covers ~13 of
  the 19 golden pairs — the exclusion is a sibling `RISCOS EXCLUÍDOS` /
  `PREJUÍZOS NÃO INDENIZÁVEIS` clause, or a nested `10.2.1`-style exclusion
  under the coverage clause.*
- **`adjacent_section`** — the exclusion is in a *different* section root of the
  same document whose page span is within `ADJACENT_SECTION_MAX_PAGE_GAP` (3)
  pages of the coverage clause's. This is for the flat / parent-less OCR
  documents (doc 4, doc 7's annexes) where every clause is its own section root,
  so `same_section` structurally cannot fire. In doc 4 the reference exclusion
  sits ≤ 1 page from its coverage clause; the ±3 window is calibrated on that.
- **`cross_reference`** — a `cláusula N` / `cláusula 10.2` token parsed from the
  coverage clause's text and resolved to a same-document clause by its `path`'s
  last numbering segment, kept if `exclusion`-typed.

A clause reachable by more than one edge is kept once, under its best link.
Candidates are ranked `cross_reference` → `same_section` → `adjacent_section`,
then by tree distance, then by page gap, then by `clause_id` — fully
deterministic, no reliance on dict order.

**Cross-references are derived here, not "during M1".** The DoD says parse
in-text cross-references during M1. M1's corpus schema is frozen and a re-parse
(OCR + LLM classification + the vision boundary pass) is disproportionate for a
pure function of text the artifact already carries, so `extract_cross_references`
runs at graph-build time. It is the production formalisation of the identical
helper `scripts/find_candidate_clauses.py` ([M2-08]) already uses for golden-set
curation — that issue's docstring names this step as its production successor.
The [M2-08] script keeps its own copy: four frozen M2 authoring scripts import
it, and refactoring them is out of proportion to this issue. Recorded as a
deviation per the repo convention ([M3-04] RRF-vs-weighted, [M1-08c]
predictions).

### The reserved-budget mechanism

`ExclusionCoRetrievalRetriever` wraps the [M3-05] reranking retriever behind the
**same** `retrieve(question, *, k, metadata_filter=None) -> list[str]`:

1. ask the base for its top `k`;
2. for the `coverage` clauses in that list, collect the linked exclusions the
   base **missed**;
3. take the `RESERVED_EXCLUSION_SLOTS` best-ranked of those and append them,
   evicting one lowest-ranked *supporting* base entry each — a supporting entry
   is one that is neither `coverage` nor `exclusion`. A coverage clause is the
   primary answer and an exclusion the base already found is the whole point of
   the step, so neither is ever evicted; if nothing supporting is left to drop,
   the injection simply stops.

With no retrieved coverage clause, no link, or every linked exclusion already in
the top-k, the output is the base ranking **unchanged** — the step is inert on
the questions it has nothing to say about.

### The `[M1-09]` per-constant decision

`RESERVED_EXCLUSION_SLOTS` (1), `ADJACENT_SECTION_MAX_PAGE_GAP` (3) and
`CROSS_REFERENCE_PATTERN` live in `exclusion_co_retrieval_config.py` as module
constants with a `config_fingerprint()`, not `.env` keys. They determine the
exact top-k a published Recall@k / exclusion-clause recall is measured on, so —
per [M1-09]'s per-constant rule — they are experimental design, like
`RERANK_CANDIDATE_DEPTH` or `SEED`; a `.env` edit must not be able to move a
published result. **This issue introduces no new `.env` key** — the analog of
`docs/RERANKING.md`'s "zero".

### The harness

`scripts/eval_retrieval.py --co-retrieval` applies the step outermost, after
`--rerank`, over the corpus the harness already loads. The report's existing
`by_question_type` breakdown (the `coverage_with_exclusion` row) and pooled
`exclusion_clause_recall` are the before/after signal — no new metric code.
`scripts/tune_exclusion_co_retrieval.py` (`make tune-exclusion-co-retrieval`)
computes the hybrid+rerank base ranking once per question, then replays
co-retrieval at each slot count as pure Python (no model calls), so the whole
sweep costs seconds.

---

## Prediction

Formed from a clause-by-clause reading of all 19 `coverage_with_exclusion`
golden pairs against the parsed corpus, before the sweep:

1. The structural edges reach the reference exclusion for **~16 of 19** pairs;
   q006 (`/18/18.2` ↔ `/13/13.3`, distant cousins), q016 (grandparent-level)
   and q018 (`condicoes-gerais/2` ↔ `susep/26`, unrelated top-level sections)
   are not structurally linkable.
2. The hybrid + rerank base already has **high** exclusion-clause recall
   (92.6%, 25/27), so co-retrieval's job is to close a *small* residual gap,
   not to carry the number.
3. `coverage_with_exclusion` Recall@10 should rise by a few points; the pooled
   exclusion-clause recall should approach but not necessarily reach 100%.
4. A larger reserved-slot count risks displacing relevant lookup answers on the
   ~80% of golden questions that are not about exclusions, so the sweep should
   show a low count winning.

---

## Outcome (2026-08-29)

### The slot sweep

`make tune-exclusion-co-retrieval`, `--filter default`, all 117 scorable
`golden-set-v1` questions. `cov+excl` is the `coverage_with_exclusion` subset
(n = 19).

| reserved slots | overall R@1 | overall R@5 | overall R@10 | overall MRR | overall nDCG@10 | exclusion recall | cov+excl R@10 | cov+excl MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 (no co-retrieval) | 65.8% | 86.9% | 91.5% | 80.6% | 82.0% | 92.6% (25/27) | 78.9% | 64.8% |
| **1** | 65.8% | 87.3% | **92.3%** | 80.6% | 82.3% | **100.0% (27/27)** | **84.2%** | 65.3% |
| 2 | 64.9% | 86.9% | 91.5% | 79.8% | 81.5% | 100.0% (27/27) | 84.2% | 65.3% |
| 3 | 61.1% | 82.2% | 86.3% | 75.9% | 77.2% | 100.0% (27/27) | 84.2% | 66.4% |

**Chosen: 1 reserved slot.** It closes the exclusion-clause recall gap
completely and lifts `coverage_with_exclusion` Recall@10 by 5.3 points **with
zero regressions** — overall Recall@10 goes *up* (91.5 → 92.3) because the two
recovered questions are in the golden set and nothing else moved. Slot count 2
buys nothing further on the subset and gives back the overall gain; slot count 3
displaces relevant lookup answers hard (overall Recall@10 −5.2 points). The
base's exclusion recall is already high enough that one reserved slot is all the
budget the residual gap needs.

### Before / after — the committed configuration (1 reserved slot)

| question_type | n | R@1 | R@5 | R@10 | MRR | nDCG@10 |
| --- | --- | --- | --- | --- | --- | --- |
| coverage_with_exclusion (before) | 19 | 23.7% | 63.2% | 78.9% | 64.8% | 61.8% |
| coverage_with_exclusion (after) | 19 | 23.7% | **65.8%** | **84.2%** | **65.3%** | **64.0%** |
| cross_document | 16 | 68.8% | 93.8% | 93.8% | 80.2% | 83.7% |
| definition | 18 | 77.8% | 83.3% | 88.9% | 81.5% | 83.3% |
| direct_lookup | 64 | 74.2% | 93.2% | 95.3% | 85.1% | 87.2% |

`cross_document`, `definition` and `direct_lookup` are **identical** before and
after — co-retrieval fires on many of those questions (they retrieve coverage
clauses too) but only ever evicts a supporting clause that was not a reference,
so the metric does not move.

| pooled metric | before | after |
| --- | --- | --- |
| exclusion-clause recall | 92.6% (25/27) | **100.0% (27/27)** |
| overall Recall@10 | 91.5% | **92.3%** |
| overall MRR | 80.6% | 80.6% |
| overall nDCG@10 | 82.0% | 82.3% |
| foreign-document rate | 0.0% | 0.0% |

### What actually changed

Exactly two questions:

- **`coverage_with_exclusion-005`** (0.50 → 1.00): the reference exclusion
  `1:clausulas-adicionais/1/1.3` ("ALÉM DAS HIPÓTESES PREVISTAS EM PREJUÍZOS
  NÃO INDENIZÁVEIS…"), a sibling of the retrieved coverage clause `.../1/1.2`,
  was ranked outside the base top-10 and is now injected.
- **`coverage_with_exclusion-004`** (0.50 → 1.00): the reference exclusion
  `15:10-danos-aos-vidros-basica/10.2/10.2.1`, a descendant of the retrieved
  coverage clause, likewise.

These are the only two exclusion references the [M3-05] base missed, and
co-retrieval recovers both.

### Prediction vs actual

| # | predicted | actual |
| --- | --- | --- |
| 1 | structural edges reach ~16/19 | correct — the two the base missed are both reached and recovered |
| 2 | co-retrieval closes a small residual gap | correct — 25/27 → 27/27, two questions |
| 3 | subset R@10 up a few points, recall approaches 100% | **beat it** — subset R@10 +5.3, exclusion recall exactly 100% |
| 4 | a low slot count wins | correct — 1 is optimal, 2 neutral, 3 harmful |

### One design iteration

The first sweep used a "budget = reserved − exclusions already present"
rule and slot count 2. It moved almost nothing: the hybrid + rerank base
already puts ≥ 1 linked exclusion in the top-10 for nearly every coverage
question, so the budget was usually already "spent" on a *different* exclusion
than the reference one, and slot 2's extra injection cost one regression
(`definition-017`). Switching to "inject the N best linked exclusions the base
**missed**, regardless of how many other links are present" and slot count 1
gave the clean result above. The intermediate state is recorded here rather than
discarded, per `MILESTONES.md`.

---

## Verdict

**Keep exclusion co-retrieval, 1 reserved slot, on the [M3-05] base.** It does
exactly what [M3-06] is for: it guarantees the exclusion that limits a retrieved
coverage clause reaches the context, structurally, rather than trusting the
reranker to have ordered it in. On `golden-set-v1` that means exclusion-clause
recall goes to 100%, `coverage_with_exclusion` Recall@10 rises 5.3 points, and
nothing regresses. For M4 the guarantee matters more than the 5 points: the
compatibility node is required to weigh retrieved exclusions against retrieved
coverage, and it can only do that for exclusions retrieval actually returned.

---

## Limitations

- **The residual `coverage_with_exclusion` gap is now entirely on the coverage
  side.** Every exclusion reference is retrieved; the six subset questions still
  at Recall@10 0.50 (`-001`, `-002`, `-010`, `-016`, `-017`, `-018`) are each
  missing the *coverage* clause, which the base retriever failed to surface and
  which co-retrieval by construction does not touch. Closing that is base-
  retrieval tuning — [M3-08]'s benchmark matrix, or a future issue.
- **q018 is not structurally linkable.** `7:condicoes-gerais-seguro-automovel-
  individual-mensal/2` and `7:susep/26` are in unrelated top-level sections,
  far apart on the page — no `same_section`, `adjacent_section` or
  `cross_reference` edge connects them. It happens that q018's exclusion is
  retrieved anyway (the base finds it), so this costs nothing here, but a
  structurally-isolated coverage/exclusion pair is a real blind spot.
- **The step fires broadly.** Insurance filings pair a coverage section with an
  exclusion section as siblings almost everywhere, so co-retrieval triggers on
  most `direct_lookup` and `definition` questions too. With 1 reserved slot and
  the supporting-only eviction rule this is provably harmless on
  `golden-set-v1` (identical metrics), but on a different question set a
  supporting clause that happens to be a reference and happens to sit at rank 10
  could be displaced. The sweep is the guardrail: re-run it if the base
  changes.
- **`clause_type` comes from the [M1-05b] LLM classifier** (99.7% of the
  corpus). A coverage clause mistyped as `condition` gets no co-retrieval; an
  exclusion mistyped as `condition` is not a candidate. This is the same
  dependency the pooled exclusion-clause-recall metric already has.

---

## Deferred / handed to later issues

- The lexical / dense / hybrid / hybrid+rerank+co-retrieval **benchmark
  matrix** and `make build-index` — done: [M3-08], `docs/RETRIEVAL_BENCHMARK.md`.
  The committed best config is this one (Recall@10 92.3%, exclusion recall 100%).
- Coverage-clause recall on the `coverage_with_exclusion` subset — base
  retrieval tuning, not re-run by [M3-08] (bars cleared); a future issue or
  golden-set-v2.
- M4's retrieval node constructs its own `ExclusionCoRetrievalRetriever` with a
  `reserved_slots` sized for its context budget; the config constant is the
  default and the golden-set-validated value.
