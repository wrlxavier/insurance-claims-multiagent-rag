"""Loader for the curated one-click example claims -- [M6-03].

``examples.json`` sits beside this module (it ships in the image via
``COPY app/src``). ``data/synthetic_claims/*.jsonl`` -- the set these were copied
from -- is deliberately *not* in the image, so the demo carries its own copy.
"""

from __future__ import annotations

import functools
from pathlib import Path

from pydantic import TypeAdapter

from presentation.demo.schemas import DemoExample

_EXAMPLES_PATH = Path(__file__).parent / "examples.json"
_ADAPTER = TypeAdapter(list[DemoExample])


@functools.cache
def load_examples() -> list[DemoExample]:
    """Return the curated examples, parsed and validated once per process.

    Raises on a malformed ``examples.json`` -- a loud failure of the demo
    endpoint only, and only on first call, never at import time.
    """
    return _ADAPTER.validate_json(_EXAMPLES_PATH.read_bytes())
