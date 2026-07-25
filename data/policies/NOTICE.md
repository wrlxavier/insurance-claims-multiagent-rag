# NOTICE

This repository redistributes third-party material. This notice states what that
material is, where it came from, and on what basis it is included.

## Scope of the MIT license

The MIT license in `LICENSE` covers **the source code of this repository only**.

It does not cover the PDF documents under `data/policies/raw/`. Those documents
were written by the insurance companies that filed them, are not licensed by this
project, and their presence here grants no license to them.

## Source of the policy documents

|                          |                                                                                     |
| ------------------------ | ----------------------------------------------------------------------------------- |
| Source                   | SUSEP — Consulta Pública de Produtos (public product registry)                        |
| Portal                   | https://www.gov.br/pt-br/servicos/consultar-produtos-susep                            |
| Retrieved                | 2026-07-25T19:00:00-03:00                                                             |
| Contents                 | 30 PDFs, Brazilian motor insurance lines (SUSEP *ramo* 05)                            |
| Per-document provenance  | `data/policies/manifest.csv`                                                          |

SUSEP (Superintendência de Seguros Privados) is Brazil's federal insurance
regulator. Its Consulta Pública de Produtos is a free public service that lets
anyone retrieve the registered conditions of an insurance product by its process
number. Every document in this corpus was obtained through that service, unmodified.

Insurer, CNPJ, SUSEP process number, filed version and retrieval date for each
individual document are recorded in `data/policies/manifest.csv`. That file, not
this notice, is the authoritative per-document record.

## Basis for inclusion

The text of each document was authored by the insurer that filed it, not by SUSEP.
Being publicly consultable through a regulator's portal is not the same as being
released for redistribution, and **this project makes no claim that these documents
are in the public domain**.

They are reproduced here in good faith, in full and unaltered, for a limited,
non-commercial, research and educational purpose: so that a reader can reproduce
the retrieval and evaluation results reported in this repository without
re-collecting the corpus by hand.

**If you hold rights in any of these documents and want one removed, write to
`wrxavier.code@proton.me` and it will be taken down promptly.**

## What these documents are — and what they are not

These are **general and special conditions of registered insurance products**
(*condições gerais*). They are product templates filed with the regulator. They
are **not** individual insurance contracts.

They therefore contain no:

- coverages actually purchased by any customer
- insured amounts, deductibles, or premiums
- policy periods, endorsements, or renewals
- personal data of any kind

SUSEP's own portal guidance is explicit on this point: what it publishes is the
complete registered plan, so it may describe coverages a given customer never
bought, because coverages that were not contracted and individually negotiated
clauses never appear in the document handed to that customer.

This is a substantive limit on what any system built from this corpus can honestly
claim. A system reading these documents can assess whether a described event is
**consistent or inconsistent with the conditions of a registered product**. It
cannot determine whether a real claim is covered under a real policy, because the
contract-level facts that determination requires are absent by construction. See
`README.md` and `docs/EVALUATION.md`.

## No personal data

This repository contains no claim, customer, or policyholder data. All claim
descriptions used for evaluation are synthetic and were generated for this project.
No real insured person's data was used at any stage.

## Trademarks

Insurer names and logos appear within the documents, and insurer names appear in
`data/policies/manifest.csv`, solely to identify the source of each document. This
project is not affiliated with, endorsed by, or sponsored by SUSEP or by any
insurer named here.

## Disclaimer

Nothing in this repository is legal advice, insurance advice, or a coverage
determination. The documents are reproduced as retrieved and may have been
superseded by later filed versions. Outputs produced by the software in this
repository are illustrative and are not decisions about any real insurance claim.