# The local demo UI — [M6-03]

A recording surface, not a product. It exists to show — on video — the one thing
that distinguishes this project: **a person approving, editing or rejecting the
system's recommendation before anything is final.** It is timeboxed and
throwaway.

## Run it

It ships inside the existing stack, no extra setup:

```bash
docker compose up -d --build     # postgres, redis, migrate, api, worker
# (build/ must hold an index — `make fetch-corpus-artifacts && make build-index`)
```

Then open **`http://localhost:8000/`** (redirects to `/demo/`). Bare-metal
(`make serve` + `make worker`) serves it at the same URL.

## What it does

A single static page (`app/src/presentation/demo/static/`) that drives the
real `/v1` assessment API:

1. **Assess a claim** — paste a narrative (or pick a prepared example),
   optionally name the SUSEP process, submit. `POST /v1/assessments`.
2. **Watch progress** — the pipeline stepper (intake → clarification → retrieval
   → assessment → recommendation → human review) lights up as nodes complete,
   read live from the LangGraph checkpoint. A full run takes about one to four
   minutes (p50 83 s / p95 250 s).
3. **Review** — the verdict, confidence, reasoning, consistency flags, and each
   citation rendered as the **full clause text with its source document and page
   span**.
4. **The human checkpoint** — Approve / Edit / Reject, with a notes field. An
   edit is recorded in the decision; the system's own verdict, prose and
   citations are never overwritten.
5. **Decided** — the analyst decision beside the system opinion, then the full
   per-node **audit trail** (`GET /v1/assessments/{id}/audit`).

## The prepared examples

`app/src/presentation/demo/examples.json` (served by `GET /demo/api/examples`) —
seven claims copied verbatim from the project's synthetic set
(`data/synthetic_claims/`), each with its SUSEP process resolved:

| Example | Path it exercises | Expected |
| --- | --- | --- |
| Glass / headlight chipped by debris | covered peril | `compatible` |
| Bodily injury to a pedestrian | RCF-A's actual purpose | `compatible` |
| Scooter on a road that bans them | a retrieved exclusion | `incompatible` |
| Side collision, event date never given | intake gap → clarification loop → exhausted | `insufficient_information` |
| Mark on the glass, cause unknown | retrieval runs, context does not settle it | `insufficient_information` |
| Rear-end damage vs a liability-only product | product/claim mismatch | `incompatible` |
| Engine fire vs a liability-only product | product/claim mismatch | `incompatible` |

**"Expected", not guaranteed.** A live run is non-deterministic and about one
claim in ten fails at intake (`docs/PERFORMANCE.md`). Re-run, or pick another
take.

## What `/demo/api/*` adds — and why it is not in `/v1`

The demo needs three things the assessment API deliberately does not carry
(`docs/API.md`). They are read-only, demo-scoped, and not part of any contract:

- `GET /demo/api/examples` — the curated claims above.
- `GET /demo/api/clause-context?clause_id=…` — joins a citation's `clause_id` to
  the full clause text, its `page_start`/`page_end` and a human-readable source
  label, from `build/parsed_clauses.jsonl`. The `/v1` citation object carries
  only a short excerpt and ids. Degrades to excerpt-only if the parsed corpus is
  not mounted.
- `GET /demo/api/assessments/{id}/progress` — a live read of the LangGraph
  checkpoint (`audit_trail` channel + `next` nodes). The `/v1` status is only
  `pending` / `running` / `awaiting_review`; this is what makes the stepper move.
  Degrades to a plain elapsed timer if the checkpoint cannot be read.

## Removing it

Everything is in one directory:

```bash
git rm -r app/src/presentation/demo/
# then revert the demo block in app/src/presentation/app.py
```
