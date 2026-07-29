# Scope

This is the canonical statement of what this project can and cannot claim.
It is written once, here, and every other surface — `README.md`,
`NOTICE.md`, `data/README.md`, `docs/DATA_SOURCES.md`, and every agent
prompt via `SCOPE_PREAMBLE` in
`app/src/infrastructure/graph/prompts/scope_preamble.py` — references this
file rather than restating it, so the constraint cannot drift the first
time a prompt or a doc gets rewritten.

## What the corpus is

The documents under `data/policies/raw/` are **general and special
conditions of registered insurance products** (*condições gerais*), filed
with SUSEP, Brazil's insurance regulator. They are product templates.

## What the corpus is not

They are **not individual insurance contracts**. By construction, they
contain none of:

- coverages actually contracted by any customer
- insured amounts, deductibles, or premiums
- policy periods, endorsements, or renewals
- personal data of any kind

A registered product's filed conditions can describe coverages a given
customer never bought, because what SUSEP publishes is the complete
registered plan, not the individually negotiated contract handed to a
specific policyholder.

## What a system built on this corpus may assert

Given a described event and a registered product's filed conditions, the
system may say whether that event is **consistent or inconsistent with the
conditions of the registered product**.

## What it may never assert

Whether a real claim is covered or denied under a real policy. The
contract-level facts that determination requires — what was actually
contracted, for how much, under which deductible, during which period —
are absent from this corpus by construction. No amount of retrieval or
reasoning recovers a fact that was never in the source documents.

This distinction is load-bearing for the whole project. It is not a
disclaimer bolted on afterward; it is the reason the verdict vocabulary
below exists.

## Verdict vocabulary

Every verdict produced anywhere in this system — prompts, structured
outputs, API schema enums, evaluation labels — uses exactly these three
values:

- `compatible`
- `incompatible`
- `insufficient_information`

These verdicts are never named, and no prompt, schema, or piece of interface
text may frame an assessment as `covered` / `denied`, or as any synonym
that reads as a real coverage or claims decision (approved, rejected,
payable, resolved, and the like). A `compatible` verdict says the described
event does not contradict the filed conditions — it does not say a claim
would be paid.

## The consistency/risk node is not a fraud detector

The consistency-checking node introduced in M4-06 flags internal
contradictions, implausible amounts, and narrative inconsistencies in a
claim description. It signals for human attention. It does not, and will
never, claim to detect fraud: the data available to this project does not
support that claim, and the method (a handful of deterministic checks plus
an LLM judging narrative coherence) does not either. Documentation and code
must describe it as a consistency-signalling component, nothing more.
