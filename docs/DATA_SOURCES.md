# Data sources and corpus selection

How the 30 policy documents in `data/policies/raw/` were chosen, what the
selection buys, and where it deviated from the original plan.

For what the documents are and what they legally are not, see
[`../data/README.md`](../data/README.md) and [`../NOTICE.md`](../NOTICE.md).

## Sources

Two SUSEP services, used for different things.

**The catalogue** — [Consulta de Produtos, dados.gov.br](https://dados.gov.br/dados/conjuntos-dados/consulta-de-produtos).
Open data listing every registered insurance product by company, CNPJ, process
number and line. Filtered to *ramo* 05 (motor), it yields **180 products across
52 insurers**. This is the sampling frame; it contains no document text.

**The documents** — [Consulta Pública de Produtos](https://www.gov.br/pt-br/servicos/consultar-produtos-susep).
Free public service that returns the filed conditions for a product, looked up by
process number. There is no full-text search: you must arrive with a list of
process numbers, which is what the catalogue provides.

All 30 documents were retrieved in a single session on **2026-07-25**.

## Sampling frame

| Line | Products available |
| --- | --- |
| Automóvel – Casco | 86 |
| Assistência e Outras Coberturas – Auto | 36 |
| Responsabilidade Civil Facultativa – Auto | 25 |
| Garantia Estendida – Auto | 18 |
| Carta Verde | 15 |
| **Total** | **180** |

Own-damage policies by indemnity basis:

| Sub-line | Products available |
| --- | --- |
| Valor Determinado e Valor de Mercado Referenciado | 71 |
| Valor de Mercado Referenciado | 8 |
| Valor Determinado | 7 |

## Selection criteria

The corpus is not a random sample. It is stratified against five requirements,
in priority order.

1. **Clause richness.** Half the corpus is own-damage (`CASCO`), because that is
   where coverage and exclusion clauses are densest and where most golden-set
   questions will live.
2. **Product/claim mismatch.** The other four lines are present specifically
   because a claim describing damage to the insured's own vehicle is
   *incompatible* with all of them. They test whether the intake agent classifies
   correctly instead of answering from the wrong document.
3. **Template diversity.** At most two documents per CNPJ, to maximise the number
   of distinct document layouts the parser has to survive.
4. **Same-insurer hard negatives.** Where a second document was taken from an
   insurer, it was chosen to share letterhead and boilerplate with the first but
   describe a different product — the case where retrieval is most likely to pull
   from the wrong document.
5. **Vintage spread.** Filing year is a proxy for layout era and for the
   regulatory framework in force. The corpus deliberately spans 2004 to 2025.

## Resulting composition

| Line | Documents |
| --- | --- |
| `CASCO` | 15 |
| `RCF-A` | 8 |
| `ASSIST` | 4 |
| `GAR.EST` | 2 |
| `CARTA VERDE` | 1 |

Own-damage by indemnity basis:

| Regime | Documents |
| --- | --- |
| `VD` (agreed value) | 6 |
| `VD+VMR` (both bases offered) | 6 |
| `VMR` (referenced market value) | 3 |

**22 distinct insurers**: 14 contribute one document, 8 contribute two. Large
national carriers (Bradesco, Porto Seguro, Azul, HDI, Zurich, Allianz, MAPFRE,
Generali, Caixa, Santander), digital insurtechs (Darwin, AKAD, KOVR), niche
underwriters (Suhai, Usebens) and affinity/embedded specialists (Assurant,
Cardif, Virginia Surety).

Filing year:

| Period | Documents |
| --- | --- |
| 2004–2009 | 4 |
| 2010–2016 | 6 |
| 2017–2021 | 6 |
| 2022–2025 | 14 |

### Same-insurer pairs

| Insurer | Documents | Years | What it tests |
| --- | --- | --- | --- |
| Porto Seguro | `CASCO` + `CASCO` | 2024, 2023 | Two indemnity regimes from one insurer — retrieval must not blend them |
| AKAD | `CASCO` + `RCF-A` | 2019, 2025 | Own-damage vs third-party liability under one brand |
| Seguros SURA | `CASCO` + `RCF-A` | 2022, 2021 | Same |
| KOVR | `CASCO` + `RCF-A` | 2022, 2024 | Same |
| Caixa Seguradora | `CASCO` + `RCF-A` | 2016, 2013 | Same |
| Bradesco Auto/RE | `CASCO` + `RCF-A` | 2014, 2014 | Same, both filed the same year |
| HDI Seguros | `CASCO` + `CARTA VERDE` | 2025, 2006 | Same insurer, 19 years apart, different lines |
| MAPFRE | `CASCO` + `ASSIST` | 2004, 2005 | Indemnity vs non-indemnity product from one insurer |

### Deliberate edge cases

**Liability-only insurers.** HDI Global (`15414.634764/2024-94`) and ARCA
(`15414.620201/2024-19`) appear in the catalogue with third-party liability
products and no own-damage product at all. A claim about the insured's own
vehicle cannot be assessed against either. These are the clearest product/claim
mismatch cases in the corpus.

**Brand collision.** HDI Seguros (CNPJ `29980158000157`) and HDI Global (CNPJ
`18096627000153`) are separate legal entities sharing a brand. The corpus holds
three documents across them, in three different lines. Disambiguation has to run
on CNPJ, not on the insurer name — code that groups by name will silently merge
them.

**Legacy layouts.** Four documents were filed between 2004 and 2006. All four
carry a usable text layer; the difficulty they pose is structural — older
typesetting, repeated headers and footers, and reading order that extraction
tools recover imperfectly — not extractability.

**OCR is required for 2 of 30 documents**, unrelated to filing vintage:
`15414604545202481.pdf` (KOVR, RCF-A, id 20, filed 2024) and
`15414618005202301.pdf` (Too Seguros, ASSIST, id 25, filed 2023). Both embed
subset fonts without a ToUnicode map, so standard text extraction returns
almost nothing (30 and 35 characters per page on average, against ~2,000+ for
every other document) even though the pages render normally and OCR recovers
their text without loss. The audit behind this number — page count, font
count, fonts with a Unicode map, extracted characters per page and a verdict
per document — is in [`TEXT_LAYER_AUDIT.md`](TEXT_LAYER_AUDIT.md), produced by
`scripts/audit_text_layer.py` and cross-checked against a second extraction
backend. The verdict is also recorded per document in the `extraction_mode`
column of `data/policies/manifest.csv`.

## Deviations from the original plan

Recorded because the plan was written before the catalogue was inspected, and
three parts of it did not survive contact with the data.

**A third indemnity regime exists.** The plan called for a balance between
`VD` and `VD+VMR`. The catalogue also contains 8 products under `VMR` alone — a
distinct indemnity basis that changes the calculation as much as the other two
do. Three were included. The balance requested between the original two was kept
at 6 and 6.

**Pure agreed-value products are scarce.** Only 7 of the 86 own-damage products
use `VD` alone, and two of those come from the same insurer. An even split within
a 15-document own-damage block would have consumed almost the entire pool and
forced the insurer selection. Six were taken, from six different insurers.

**The planned atypical case was wrong.** MAPFRE was expected to be a
liability-only insurer. It is not: it holds products in all five lines, including
own-damage (`15414.100326/2004-83`). The liability-only insurers in the catalogue
are AIG, ARCA, ARUANA, HDI Global, XL and Too Seguros; ARCA and HDI Global were
selected. MAPFRE remains in the corpus for its 2004/2005 filings.

**Insurer count exceeds the original target.** The plan asked for roughly 15–20
distinct insurers; the corpus has 22. With a two-document cap and 30 slots, 15 is
the arithmetic floor, and every additional insurer adds a document template —
which is the point of the cap. The eight pairs supply the same-insurer hard
negatives that a one-per-insurer corpus would lose.

## Known limitations

**Group-level concentration.** The two-document cap is enforced per CNPJ, not per
economic group. Porto Seguro and Azul are separate legal entities in the same
group, giving that group three documents; HDI Seguros and HDI Global give the HDI
brand three. Both were kept knowingly — the HDI case is useful, the Porto case is
incidental.

**Motor lines only.** The corpus covers *ramo* 05 exclusively. Nothing here
supports claims about property, life or health insurance.

**Registered products, not contracts.** The most consequential limitation, and
the one that constrains what the system may assert. Spelled out in
[`../data/README.md`](../data/README.md).

**Author-curated evaluation.** The golden set is curated by the same person who
built the system being evaluated. Mitigations are documented in
[`EVALUATION.md`](EVALUATION.md).

## Reproducibility

Per-document provenance — insurer, CNPJ, process number, filed version, retrieval
timestamp and the reason each document was selected — is recorded in
`data/policies/manifest.csv`.