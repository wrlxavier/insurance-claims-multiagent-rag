# Clarification loop

The [M4-03] loop: when intake ([M4-02]) leaves `ClaimState.missing_information`
non-empty, the graph asks the claimant a specific question per gap and routes
back to intake, rather than proceeding on a guess. The loop is capped — a claim
that can never be completed terminates cleanly as *insufficient information*
with its gaps listed, it does not spin. The code lives at
`app/src/infrastructure/graph/build.py` (the router + `MAX_CLARIFICATION_ROUNDS`
+ the graph wiring), `app/src/infrastructure/graph/nodes/clarification.py` (the
question generator), `app/src/infrastructure/graph/nodes/clarification_exhausted.py`
(the terminal marker) and `app/src/infrastructure/graph/prompts/clarification.py`
(prompt + fallback templates).

**Scope.** Questions are *generated* here. They are *answered* by a human at the
checkpoint ([M4-09]) — in a deployment the answers are appended to the narrative
and intake re-runs. They are *consolidated into a verdict* by the recommendation
node ([M4-08]), which reads `clarification_exhausted` and maps it to
`Verdict.INSUFFICIENT_INFORMATION`. This issue neither answers questions nor
produces a verdict; with no human in the loop (the test harness, the eval) an
incomplete claim always runs to the cap.

---

## Topology

```
START -> intake -> route_after_intake
    "proceed"       -> END          (M4-04 retargets this to the retrieval node)
    "clarification" -> clarification -> intake      (loop)
    "exhausted"     -> clarification_exhausted -> END  (M4-08 retargets this to the recommendation node)
```

`route_after_intake` (`build.py`) runs after every intake pass:

| `missing_information` | `clarification_rounds` | route |
|---|---|---|
| empty | — | `"proceed"` |
| non-empty | `< MAX_CLARIFICATION_ROUNDS` | `"clarification"` |
| non-empty | `>= MAX_CLARIFICATION_ROUNDS` | `"exhausted"` |

- **`MAX_CLARIFICATION_ROUNDS = 2`** — "ask, then follow up once", the realistic
  patience budget for an intake bot before escalating to a human. It is a code
  constant (like the retry constants in
  `application/use_cases/llm_retry_defaults.py`), not an `.env` knob: it is
  product behaviour, tested in code, not a deployment setting.
- Termination is a property of this router plus the cap. The graph never relies
  on LangGraph's `recursion_limit` / `GraphRecursionError` — the worst case is
  `intake ×3 + clarification ×2 + clarification_exhausted ×1`, well under the
  default limit of 25.

### The `clarification` node

One `fast_model` call (structured output into `schemas.ClarificationOutput`),
one `ClarificationQuestion` per gap in the current `missing_information` list,
appended to the accumulated `clarification_questions` channel; the node also
increments `clarification_rounds` (it *is* the event "a round happened").

Unlike intake, a question generator has a sane fallback: any gap the model does
not address, and every gap when the LLM call fails all its retries, is filled
from `CLARIFICATION_FALLBACK_TEMPLATES` (one Brazilian-Portuguese question per
`MissingInfoTag`). So `len(questions this round) == len(missing_information)`
always, and a provider outage cannot break loop termination.

### Preserving context across iterations

On a re-entry the intake node ([M4-02]) does **not** restart from zero:

- it feeds the questions already put to the claimant into its prompt, so it does
  not simply re-flag a gap the claimant was already asked about;
- it merges this pass's extraction over the previous `entities` — a fresh
  non-null value wins, a previous non-null value fills any field this pass left
  null. An already-extracted fact is never lost when extraction re-runs.

`missing_information` is recomputed fresh each pass (last-write-wins, as the
state schema intends) so an answered gap drops off the list.

### The `clarification_exhausted` node

Deterministic, no model. Sets `clarification_exhausted = True`; the open gaps
stay in `missing_information`. This marker is kept distinct from
`context_sufficient` (owned by [M4-04] / the [M3-07] retrieval gate) so [M4-10]
can catalogue "the claimant never supplied enough" separately from "retrieval
missed".

---

## Method

`make eval-clarification` (`scripts/eval_clarification.py`) runs two passes over
the 13 `insufficient_information` claims in
`data/synthetic_claims/claims.jsonl` with the real fast model and writes
`eval/runs/clarification_loop.{md,json}` plus a per-claim
`clarification_questions.jsonl`. The committed numbers below are one such run.

- **Termination pass** — one full compiled-graph invocation per claim (real fast
  model for both intake and clarification). Confirms every claim terminates and
  reports how many actually entered the loop (intake's `missing_information`
  recall is imperfect — see `docs/INTAKE_EXTRACTION.md`) and the
  `clarification_rounds` distribution.
- **Question-quality pass** — feeds each claim's labelled `missing_fact_type`
  straight in as `missing_information` and calls the clarification node once, so
  there is exactly one generated question per claim to inspect for DoD item 2
  ("a specific question per missing field, not a generic request for more
  detail"). A crude regex flags generic phrasing (`mais detalhes`,
  `mais informações`, `poderia detalhar`, …).

Structural loop termination — the DoD's "verify the loop terminates in every
case" — is proven with a fake LLM in
`tests/unit/infrastructure/graph/test_claim_graph.py`
(`test_every_incomplete_m2_04_claim_terminates`), which runs in CI. This live
eval is the on-demand quality check, not the termination guarantee.

---

## Results

One `make eval-clarification` run, `deepseek/deepseek-v4-flash-0731` via the
OpenRouter `baidu/fp8` route (same model + route as intake), 13 claims, ~7.5 min
wall clock, 0 errors. Full output: `eval/runs/clarification_loop.md` +
`eval/runs/clarification_questions.jsonl`.

**Termination (full compiled graph, real fast model both nodes).**

- **13/13 terminated** without hanging.
- 6/13 actually entered the loop live — for the other 7, intake did not re-flag
  the deliberately-omitted fact, so the claim took the `"proceed"` branch. That
  tracks the M4-02 pre-retrieval recall frontier (`docs/INTAKE_EXTRACTION.md`):
  only `data_evento_vigencia` is reliably flagged; `valor_franquia_limite` and
  `tipo_evento_condicao` narratives read like answerable ones. The gaps those 7
  carry are caught downstream (retrieval / the M3-07 gate / the assessment node
  returning `insufficient_information`), not here.
- `clarification_rounds` distribution: `{0: 7, 2: 6}` — every claim that entered
  the loop ran to the cap (expected: the harness has no human to answer, so
  intake keeps re-flagging).

**Question quality (one forced clarification call per claim).**

- **13/13** questions generated, one per gap.
- **0/13** flagged as generic phrasing.
- Question length: min 117 chars, mean 192 — every question names the specific
  fact and refers back to the claimant's own narrative. Examples:
  - `data_evento_vigencia` → *"Sobre o acidente na garagem do condomínio: me
    confirma a data e o horário exatos em que a sua esposa bateu o carro?
    Preciso disso pra verificar se a apólice estava valendo nesse dia."*
  - `tipo_evento_condicao` → *"Você disse que não sabe se alguém bateu ou se
    caiu algo de uma árvore. … como tá essa marca no vidro? É um trincado, um
    furo, um risco comprido? E no estacionamento, tinha pedra, galho caído …?"*
  - `ambito_geografico` → *"Você falou que o carro apagou numa reta da rodovia
    durante a viagem de férias. Me conta: em qual cidade e estado isso
    aconteceu, ou pelo menos qual rodovia e trecho vocês estavam?"*

The DoD bar — "a specific question per missing field, not a generic request" —
is met: the model consistently anchors each question to the narrative, and the
template fallback (never triggered in this run) is field-specific by
construction.
