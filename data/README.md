# Data

Everything this project reads from disk. Three things live here: the policy
corpus, the golden set used to measure retrieval and end-to-end quality, and the
synthetic claims used as test inputs.

## Read this first

The documents under `policies/raw/` are **general and special conditions of
registered insurance products** (*condições gerais*). They are product templates
filed with the Brazilian regulator. **They are not individual insurance
contracts.**

They contain no contracted coverages, no insured amounts, no deductibles, no
policy periods, no endorsements and no personal data — none of it, by
construction. A system reading this corpus can decide whether a described event
is **consistent or inconsistent with the conditions of a registered product**. It
cannot decide whether a real claim is covered under a real policy. That
distinction is load-bearing for this project and is reflected in the wording of
every agent prompt and every evaluation label.

Provenance, rights and takedown contact: [`../NOTICE.md`](../NOTICE.md).
Selection methodology: [`../docs/DATA_SOURCES.md`](../docs/DATA_SOURCES.md).

## Layout

```
data/
├── policies/
│   ├── raw/            30 PDFs, exactly as retrieved — write-once, never edited
│   └── manifest.csv    one row per document; the authoritative record
├── golden_set/         evaluation questions with reference clauses
└── synthetic_claims/   generated claim descriptions used as test inputs
```

`policies/raw/` is immutable. It is the only directory in the project that does
not change after the data-collection phase. Everything downstream — parsed
clauses, embeddings, the vector index — is derived from it, is reproducible with
`make build-index`, and is therefore not tracked in git.

## `manifest.csv`

One row per document, 30 rows. Columns marked *verbatim* are copied unchanged
from SUSEP's open product catalogue so the corpus can be joined back to it.

| Column | Description |
| --- | --- |
| `id` | Sequential 1–30. Stable; used to refer to documents in discussion and in `docs/`. |
| `filename` | File under `policies/raw/`. Always `susep_process` with punctuation stripped, plus `.pdf`. |
| `product_line` | Insurance line, abbreviated. See vocabulary below. |
| `indemnity_regime` | How indemnity is calculated. Applies to `CASCO` only; the literal `n/a` elsewhere. See vocabulary below. |
| `insurer` | *Verbatim.* Legal name of the filing company, in Portuguese. |
| `cnpj` | *Verbatim (zero-padded — see below).* Brazilian company registration number, 14 digits. The reliable identity key: two companies can share a brand and differ here. |
| `susep_process` | *Verbatim.* SUSEP process number, `NNNNN.NNNNNN/NNNN-NN`. Identifies the registered product, not a single document. |
| `process_year` | Year embedded in `susep_process`. Derived; a proxy for document vintage and layout era. |
| `susep_ramo` | *Verbatim.* Full SUSEP line description, in Portuguese. |
| `susep_subramo` | *Verbatim.* Full SUSEP sub-line description, in Portuguese. Source of `indemnity_regime`. |
| `susep_version` | Which filed version of the product was downloaded. See "Version pinning" below. |
| `version_start_date` | Start of that version's commercialisation period, `YYYY-MM-DD`. |
| `page_count` | Pages in the PDF. Filled by `scripts/verify_corpus.py --write`. |
| `retrieved_at` | Download timestamp, ISO 8601 with UTC offset. Constant across the corpus: one collection session. |
| `source_url` | Portal the documents were retrieved from. |
| `selection_rationale` | Why this document is in the corpus. Authored, not sourced. |

### Vocabulary

`product_line` and `indemnity_regime` keep their Portuguese abbreviations
deliberately: translating them would break the join back to SUSEP and add a
mapping layer to maintain. They are identifiers, not interface text.

| Code | SUSEP line | What it covers |
| --- | --- | --- |
| `CASCO` | Automóvel – Casco | Damage to the insured's own vehicle: collision, fire, theft, total loss |
| `RCF-A` | Responsabilidade Civil Facultativa – Auto | Optional third-party liability — damage the insured causes to others |
| `ASSIST` | Assistência e Outras Coberturas – Auto | Roadside assistance and ancillary services; not an indemnity product |
| `GAR.EST` | Garantia Estendida – Auto | Extended warranty: mechanical/electrical failure after the manufacturer's warranty |
| `CARTA VERDE` | Carta Verde | Mandatory cross-border cover for travel within Mercosur countries |

| Code | SUSEP sub-line | Indemnity basis |
| --- | --- | --- |
| `VD` | Valor Determinado | Agreed value: a fixed amount set at underwriting |
| `VMR` | Valor de Mercado Referenciado | Referenced market value at the time of loss, via a published price index |
| `VD+VMR` | Valor Determinado e Valor de Mercado Referenciado | Product offers both bases; the contract selects one |
| `n/a` | — | Not an indemnity product |

The four non-`CASCO` lines exist in the corpus on purpose: a claim describing
damage to the insured's own vehicle is **incompatible** with all of them. They
are the product/claim mismatch cases the intake agent has to classify correctly
rather than answer from the wrong document.

### Version pinning

A SUSEP process number identifies a registered product, and a product can have
several filed versions with different commercialisation periods. `susep_process`
alone therefore does not identify a document.

Rule used for this corpus: **the version the portal presented as current on
`retrieved_at` was taken for every document.** Where that rule was not followed,
`susep_version` and `version_start_date` record the version actually downloaded.

### Known upstream defect: CNPJ leading zeros

SUSEP's open product catalogue publishes CNPJ as a number rather than a
fixed-width string, so 41 of its 180 motor-line rows have lost a leading zero and
carry 13 digits instead of 14. The defect is in the source, not introduced here.

CNPJ is fixed at 14 digits, so the correction is deterministic: left-pad with
zeros.

## Not tracked in git

Parsed clauses, embeddings, the vector index and every other derived artefact.
They change on every iteration and are rebuilt from `policies/raw/`. See
`.gitignore`.