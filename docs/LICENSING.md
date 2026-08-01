# Licensing

## Scope

This document covers the licenses of **third-party dependencies** used by
this project's source code. It does not cover the PDF policy corpus; that
is [`NOTICE.md`](../NOTICE.md) and [`data/policies/NOTICE.md`](../data/policies/NOTICE.md).

## PyMuPDF (`fitz`)

The text-extraction pipeline (M1-01) uses [PyMuPDF](https://pymupdf.readthedocs.io/)
to read the PDF text layer (`get_text("dict")`), because it returns page
number, bounding box and font size per span in a single call — exactly the
raw material heading and clause-tree recovery (M1-04) need.

PyMuPDF is dual-licensed: **AGPLv3**, or a commercial license from its
publisher, Artifex. This project uses it under **AGPLv3**. That is a
deliberate, accepted choice for this project in its current form: an
offline batch pipeline (`make extract-text`) run locally or in CI, not a
network-accessible service. AGPLv3's defining obligation — that users
interacting with the software over a network must be offered the
corresponding source — does not bite under that usage pattern.

**If this project is ever deployed as a network-accessible service** (an
API, a hosted demo, etc.) whose request path touches PyMuPDF-derived code,
this needs to be reassessed: either the AGPLv3 network-use clause must be
honored (offer source to users of that service), or the commercial license
must be obtained instead.

PyMuPDF is isolated behind the `TextExtractor` port
(`app/src/application/ports/text_extractor.py`); only
`app/src/infrastructure/parsing/extraction.py` imports `fitz` directly, and
`tests/architecture/test_layer_boundaries.py` enforces that `domain` and
`application` never do. This isolation is standard Clean Architecture
practice for swappable infrastructure, not something added specifically to
route around the license.

## Everything else

All other current dependencies (`pydantic`, `pydantic-settings`, `pypdf`,
`pyarrow`) are permissively licensed (MIT/Apache-2.0/BSD family) and impose
no obligations beyond attribution already satisfied by this repository's own
`LICENSE`.
