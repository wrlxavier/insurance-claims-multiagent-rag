# Adversarial prompt-injection fixtures ([M5-08])

Two files in this directory, for two different consumers:

- `fixtures.jsonl` — adversarial probes for the real compatibility node
  (`make eval-prompt-injection`), documented below.
- `benign_imperative_clauses.jsonl` — real, non-adversarial imperative
  clause excerpts for the optional runtime classifier's false-positive
  benchmark (`make eval-prompt-injection-classifier`, [M5-08 Appendix]),
  documented at the bottom of this file.

## `fixtures.jsonl`

`fixtures.jsonl` — hand-authored adversarial probes for
`scripts/eval_prompt_injection.py` (`make eval-prompt-injection`). Unlike
`data/synthetic_claims/` and `data/golden_set/`, these are **not** produced by
the deterministic-selection + LLM-phrasing + human-review pipeline
(`data/synthetic_claims/README.md`): that pipeline samples a representative
distribution, and these fixtures are the opposite of representative on
purpose — each one deliberately salts a clause excerpt or a claim narrative
with an imperative instruction trying to hijack the verdict. Representative
sampling would dilute exactly the cases this eval exists to probe.

## Two kinds

- **`clause_injection`** — one `claim_narrative` plus a fixed `citations` list
  (built directly, not through retrieval) where a clause excerpt carries an
  injected instruction. `expected_verdict` is what the *un-injected* clause
  content settles, regardless of what the injection asks for.
- **`claim_injection`** — `claim_narrative_clean` and `claim_narrative_injected`
  over the identical `citations`. `expected_verdict` must hold for both; the
  eval additionally asserts the two narratives produce the *same* verdict.

## Fields

`schema_version`, `fixture_id`, `kind`, `entities` (an `ExtractedEntities`-shaped
object), `citations` (a list of `Citation`-shaped objects — `clause_id`,
`document_id`, `susep_process`, `clause_type`, `relevance_score`, `excerpt`),
`claim_narrative` (`clause_injection`) or `claim_narrative_clean` /
`claim_narrative_injected` (`claim_injection`), `expected_verdict`, `notes`.

`document_id` values (`doc-synth-*`) are synthetic and never resolve to a real
corpus document — these fixtures bypass retrieval entirely, so nothing here
is looked up against `data/policies/`.

## Why compatibility only

`scripts/eval_prompt_injection.py` runs the real
`infrastructure.graph.nodes.compatibility.compatibility` node (the one node
that produces a verdict from clause text). `recommendation` and `consistency`
have no verdict field to hijack ([M4-08], [M4-06]) — a prompt injection there
could at most leak into free prose, not into a decision this system reports.

## `benign_imperative_clauses.jsonl` ([M5-08 Appendix])

Ten hand-picked, **verbatim** excerpts from the real parsed corpus
(`build/parsed_clauses.jsonl`), each carrying `document_id` / `clause_id` /
`susep_process` / `insurer` for traceability back to its source PDF. None of
this text is adversarial or synthetic — every row is genuine SUSEP policy
wording, chosen because it uses the same imperative verb forms an injected
instruction would ("o segurado obriga-se a...", "fica vedado...", "deverá...")
while meaning nothing of the kind. `scripts/eval_prompt_injection_classifier.py`
scores these as the false-positive side of the Appendix's domain benchmark;
the four rows in `fixtures.jsonl` (their injected spans) are the detection
side. Schema: `infrastructure.evaluation.benign_clause_schema.BenignClauseFixture`.
