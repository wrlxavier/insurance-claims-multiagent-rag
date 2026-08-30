# Architecture

Cross-cutting design decisions that shape the pipeline and are not owned by any
one stage's document. Each entry states the decision, why it is a deliberate
choice rather than an incidental one, and where the evidence lives.

Stage-local rationale stays in the stage's own document — `docs/PARSING.md`,
`docs/EMBEDDINGS.md`, `docs/LEXICAL_RETRIEVAL.md`, `docs/HYBRID_RETRIEVAL.md`,
`docs/RERANKING.md`, `docs/EXCLUSION_CO_RETRIEVAL.md`. The project scope
statement is `docs/SCOPE.md`.

---

## Exclusion co-retrieval is a domain rule, not a retrieval heuristic — [M3-06]

**Decision.** After the retrieval pipeline ranks, for every retrieved `coverage`
clause the system pulls the `exclusion` clauses structurally linked to it — a
sibling or nested exclusion in the same section, an exclusion in an adjacent
section of a flat document, or one named by an in-text `cláusula N` cross-
reference — and reserves a slot of the final context for the best-linked
exclusion the ranking missed. `ExclusionCoRetrievalRetriever` and `ClauseGraph`
in `app/src/infrastructure/rag/exclusion_co_retrieval.py`; the tuned constants
and the fingerprint in `exclusion_co_retrieval_config.py`.

**Why it is a rule and not a heuristic.** A coverage clause tells you an event
*is* covered; the exclusion three paragraphs down tells you it *is not*, under
the circumstances that actually apply. An assessment built on the first without
the second is fluent, cites a real clause id, and reaches the wrong verdict —
the single failure mode M3 was scoped around ("a retriever that returns the
first without the second is worse than one that returns nothing"). The linked
exclusion is therefore not "a nice extra result" whose value a relevance score
should decide; it is **as load-bearing as the coverage clause it modifies**, and
the pipeline is not permitted to return one without the other when a structural
link exists. Reranker scores order passages by topical similarity to the
question — a question phrased as a coverage question will score the exclusion
lower — so leaving this to the ranker is leaving it to the wrong signal. The
step runs as deterministic Python over the M1 clause tree, in the same spirit as
M4-06's deterministic consistency checks: a structural fact the system should
not need a model's permission to act on.

**What it costs.** The step fires on most questions that retrieve a coverage
clause, not only exclusion questions, because insurance filings pair coverage
and exclusion sections as siblings almost everywhere. With one reserved slot and
an eviction rule that only ever drops a *supporting* clause (never a coverage or
an exclusion clause), this is measurably free on `golden-set-v1`: exclusion-
clause recall 92.6% → 100%, `coverage_with_exclusion` Recall@10 78.9% → 84.2%,
every other question type unchanged, overall Recall@10 91.5% → 92.3%. The slot
count, the adjacent-section page window and the "inject what the base missed"
rule were all set from the sweep in `docs/EXCLUSION_CO_RETRIEVAL.md`, which also
records the one design iteration and the residual limitations (a
structurally-isolated coverage/exclusion pair is a blind spot; the remaining
subset gap is now entirely coverage-side).

**Deviation on record.** The DoD asks for in-text cross-references to be parsed
during M1. M1's corpus schema is frozen and a re-parse is disproportionate for a
pure function of text the artifact already carries, so `extract_cross_references`
runs at graph-build time — the production formalisation of the helper
`scripts/find_candidate_clauses.py` ([M2-08]) already uses for golden-set
curation. Full rationale in `docs/EXCLUSION_CO_RETRIEVAL.md`.
