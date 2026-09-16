"""The local demo UI -- a recording surface, not a product [M6-03].

A vanilla static single-page app (``static/``) served by the same FastAPI app as
the assessment API, plus three read-only helper endpoints (``router.py``) the SPA
needs and the ``/v1`` contract deliberately does not carry:

- ``GET /demo/api/examples`` -- the curated one-click synthetic claims;
- ``GET /demo/api/clause-context`` -- joins a citation's ``clause_id`` to the
  full clause text, its page span and a human-readable source label, from
  ``build/parsed_clauses.jsonl`` (the ``/v1`` citation object has none of that);
- ``GET /demo/api/assessments/{id}/progress`` -- a live read of the LangGraph
  checkpoint so the pipeline stepper advances as nodes complete.

Everything for the feature lives under this one directory on purpose: it is
timeboxed and throwaway, and removing it is ``git rm -r`` here plus reverting one
hunk in ``presentation/app.py``. This is a deliberate deviation from the
``presentation/routes/`` convention -- chosen for removability.

See ``docs/DEMO_UI.md``.
"""

from pathlib import Path

DEMO_STATIC_DIR = Path(__file__).parent / "static"
"""The directory ``presentation.app`` mounts at ``/demo`` as ``StaticFiles``."""
