# Intake extraction

The [M4-02] node: it turns a free-text claim narrative
(`ClaimState.raw_claim_text`) into a structured `ExtractedEntities` record,
classifies the event against the corpus's five product lines, and populates
`ClaimState.missing_information` — the list the clarification loop ([M4-03])
consumes. The code lives at `app/src/infrastructure/graph/nodes/intake.py`
(node), `app/src/infrastructure/graph/schemas.py` (`IntakeOutput`) and
`app/src/infrastructure/graph/prompts/intake.py` (prompt); this document is the
method and the numbers.

**Why measure it separately.** Parsing quality bounds retrieval quality;
extraction quality bounds everything the graph does after intake. A claim
classified into the wrong product line pre-filters retrieval to the wrong
document ([M4-04]); a missing fact intake fails to flag is a fact the
clarification loop never asks for and the assessment node then guesses at. The
end-to-end verdict accuracy is [M4-10]'s; this is the upstream number.

**Scope.** This issue builds the node and measures its extraction over the
synthetic claim set. It does **not** wire the graph (edges are [M4-07], the
checkpointer [M4-09]), add the clarification loop ([M4-03]), or call retrieval
([M4-04]). `missing_information` is *populated* here; *acted on* there.

---

## Method

`make eval-intake` (`scripts/eval_intake.py`) runs the node once per claim with
the real fast model and writes `eval/runs/intake_extraction.{md,json}` plus a
per-claim `intake_extraction_predictions.jsonl`. The committed numbers below are
one such run.

- **Inputs.** All 51 synthetic claims: 40 `data/synthetic_claims/claims.jsonl`
  + 11 `data/synthetic_claims/product_claim_mismatch.jsonl`. ([M4-02]'s DoD says
  "30 synthetic claims"; the finalised set has since grown to 51 — all are
  scored.)
- **Model.** `deepseek/deepseek-v4-flash-0731` via the OpenRouter `baidu/fp8`
  route (`LLM_MODEL_FAST` / `LLM_FAST_PROVIDER_ORDER`, no fallback), the same
  model+route the corpus classifier uses.
- **Run cost.** ~0.22M tokens, ~12 min wall clock for 51 sequential calls.

**What is ground truth and what is not.** The claim rows carry no reference
value for the free-text entity fields, so `event_type` / `event_date` /
`description` / `estimated_amount` / `vehicle_info` are reported as
population rates (how often non-null), not accuracy. Two labels *are* ground
truth:

- the target document's `product_line` (`data/policies/manifest.csv`). For a
  `compatible` claim the described event belongs to that line by construction,
  so that cohort is the cleanest classification signal. An `incompatible`
  claim often describes an event from a *different* line — that is *why* it is
  incompatible — so a "wrong" line there can be the correct read; it is
  reported but not used as the headline.
- `missing_fact_type` on the 13 `insufficient_information` claims (one of
  `ambito_geografico` / `uso_do_veiculo` / `data_evento_vigencia` /
  `valor_franquia_limite` / `tipo_evento_condicao`) — the fact the narrative
  was deliberately written to omit. `missing_information` should contain it.

---

## Results

Run of 2026-08-31, `deepseek-v4-flash-0731` / `baidu/fp8`. 51/51 claims
processed, 0 exceptions.

### Product-line classification

| cohort | accuracy |
| --- | --- |
| `compatible` claims (event belongs to the target line) | **71%** (10/14) |
| `insufficient_information` claims | 77% (10/13) |
| `incompatible` claims (event may belong elsewhere) | 69% (9/13) |
| product/claim mismatch → `CASCO` | **100%** (11/11) |
| null classification (declined to place) | 1/51 |

Confusion, event line → predicted:

| target line | result |
| --- | --- |
| CASCO (23) | 22 CASCO, 1 null |
| RCF-A (9) | 9 RCF-A |
| GAR.EST (6) | 6 GAR.EST |
| CARTA VERDE (4) | 3 CARTA VERDE, 1 RCF-A |
| ASSIST (9) | 5 CASCO, 4 GAR.EST — **0 correct** |

The **product/claim mismatch set is the one that matters for compliance**: 11
narratives describing damage to the insured's own vehicle, each aimed at a
non-CASCO product. Intake classified every one as `CASCO`, so [M4-04] will
retrieve against own-damage conditions and the assessment will reach
`incompatible` rather than answering from the wrong product's clauses.

### `missing_information`

| | value |
| --- | --- |
| `missing_fact_type` recall (13 `insufficient_information` claims) | **54%** (7/13) |
| exact-list match | 46% |
| false-positive rate (38 answerable claims) | **8%** |

Per tag: `data_evento_vigencia` **5/5**, `tipo_evento_condicao` 1/2,
`valor_franquia_limite` 1/5, `ambito_geografico` 0/1 (`uso_do_veiculo` has no
support in the set).

### Entity extraction

Population (non-null rate), and the two omission cross-checks:

| field | non-null | note |
| --- | --- | --- |
| `event_type` | 100% | |
| `description` | 100% | |
| `product_line` | 98% | |
| `event_date` | 90% | null on **5/5** of the `data_evento_vigencia` claims — the omission is reflected, not hallucinated |
| `vehicle_info` | 92% | |
| `estimated_amount` | 4% | null on **5/5** of the `valor_franquia_limite` claims; narratives almost never state a figure |
| `susep_process` | 0% | **0 invented** — no narrative states one, and the node fabricated none (DoD item 4) |

---

## Findings

**1. ASSIST is not classifiable from the narrative alone.** The nine
ASSIST-targeted claims describe a breakdown or an incident ("o carro deu um
defeito do nado", "aluguei um carro… uma pedra trincou o vidro") and ask, in
effect, for help. Nothing in the text says whether the person wants a tow, a
warranty repair, or indemnification — the distinction that separates ASSIST from
GAR.EST and CASCO. This is the documented edge case, not the normal flow: when
the claim references a known policy (the usual input, per `docs/SCOPE.md`) the
product line is already fixed and intake's classification is a cross-check, not
the source of truth. The synthetic claims deliberately omit the insurer, so
they exercise exactly the case the classifier is weakest on.

**2. `missing_information` beyond dates needs context intake does not have.**
`data_evento_vigencia` is a crisp rule ("no time reference at all") and intake
gets it every time. `valor_franquia_limite` is not: the five claims that omit a
load-bearing amount are linguistically indistinguishable from the ~18 answerable
damage claims that also omit one — only the specific gating clause (unknown
before retrieval) tells them apart. The prompt is tuned toward **low
false-positives** (8%) so the clarification loop does not interrogate every
claimant; the cost is ~50% recall on the amount/cause/location tags. Those gaps
are caught downstream — by retrieval returning nothing usable ([M4-04] / the
[M3-07] gate) or by the assessment node returning `insufficient_information`.

**3. Entity extraction and the no-invention rule hold.** Every claim produced an
`event_type` and a `description`; the model never invented a date, an amount, or
a policy identifier where the narrative gave none, and the null fields line up
exactly with the deliberately-omitted facts.

**4. Run-to-run variance is real.** Repeated runs move the `compatible`
product-line cohort by ~1–2 claims (7 percentage points) on the same prompt.
The numbers here are one run; `make eval-intake` re-measures.

## What this means downstream

- **[M4-03]** reads `missing_information`. `data_evento_vigencia` is reliable
  input; the other tags are best-effort, and the loop's cap + the assessment's
  `insufficient_information` path are the real safety net for a fact intake
  missed.
- **[M4-04]** builds the retrieval pre-filter from `product_line` (and
  `susep_process` when present). It should treat `product_line` as a hint, not a
  hard filter, for the unidentified-policy path — the ASSIST result above is why.
- **[M4-10]** measures whether these upstream imperfections actually move the
  end-to-end verdict, which is the number that decides M4.
