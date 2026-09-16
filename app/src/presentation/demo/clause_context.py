"""The ``clause_id`` -> full-clause index behind the demo's citation cards -- [M6-03].

The ``/v1`` citation object carries a short ``excerpt`` and ids, never the whole
clause text or a page number. Both survive in ``build/parsed_clauses.jsonl`` --
keyed by ``clause_id`` in the same ``<document_id>:<path>`` form a citation uses,
already bind-mounted into the ``api`` container and already parsed at startup by
``infrastructure.rag.retriever_factory``. This module reads that file once, lazily,
and serves lookups from an in-memory dict.

It never raises on a missing or unreadable corpus: an absent index is a valid
state (the SPA degrades citations to excerpt-only), not an error.
"""

from __future__ import annotations

import functools
import logging

from presentation.demo.schemas import ClauseContext

logger = logging.getLogger(__name__)


@functools.cache
def clause_index() -> dict[str, ClauseContext]:
    """Return the ``clause_id`` -> [ClauseContext] map, or ``{}`` if unavailable.

    Cached for the process. A call that fails to read the corpus is not cached
    (``functools.cache`` only stores a returned value), so the index self-heals
    if ``build/parsed_clauses.jsonl`` appears later. Concurrent first calls may
    both parse the file -- harmless, the load is idempotent.
    """
    # Imported here, not at module load: the demo package must stay importable
    # even where the parsing package's optional deps are not installed.
    from infrastructure.parsing.corpus_artifact import (
        JSONL_PATH,
        read_parsed_clauses_jsonl,
    )

    if not JSONL_PATH.exists():
        logger.warning(
            "demo: %s absent -- citation cards will show the excerpt only "
            "(run `make fetch-corpus-artifacts`)",
            JSONL_PATH,
        )
        return {}

    try:
        rows = read_parsed_clauses_jsonl(JSONL_PATH)
    except Exception:
        logger.warning("demo: could not read %s", JSONL_PATH, exc_info=True)
        return {}

    return {
        row.clause_id: ClauseContext(
            clause_id=row.clause_id,
            document_id=row.document_id,
            title=row.title,
            text=row.text,
            page_start=row.page_start,
            page_end=row.page_end,
            clause_type=row.clause_type.value,
            susep_process=row.susep_process,
            insurer=row.insurer,
            product_line=row.product_line,
            filing_year=row.filing_year,
        )
        for row in rows
    }
