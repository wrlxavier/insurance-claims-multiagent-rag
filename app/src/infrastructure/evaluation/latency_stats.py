"""Nearest-rank latency percentiles, shared by every measurement script.

The formula -- ``sorted[min(int(n * fraction), n - 1)]`` -- was copy-pasted as a
private ``_percentile`` in ``scripts/benchmark_ann_index.py``,
``scripts/tune_reranking.py`` and ``scripts/check_embedding_input_length.py``
(and ``scripts/benchmark_retrieval_matrix.py`` via a re-import), each with the
same comment pointing at the others. [M5-10] needs it too -- for end-to-end and
per-node assessment latency -- so it lives here once.

Pure functions, no I/O, the same shape as the other
``infrastructure.evaluation`` metric modules. ``application.use_cases.chunking``
keeps its own inline copy: the application layer must not import from
``infrastructure``.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence


def percentile[NumberT: (int, float)](
    values: Sequence[NumberT], fraction: float
) -> NumberT:
    """The nearest-rank ``fraction`` percentile of ``values``.

    Accepts unsorted input (it sorts a copy). The element type is preserved --
    an all-``int`` sequence returns an ``int`` -- so a caller reporting integer
    token counts keeps integers. Raises on an empty sequence: a percentile of
    nothing is undefined, and every caller already guards the empty case with
    its own zero-valued summary.
    """
    if not values:
        raise ValueError("percentile of an empty sequence is undefined")
    ordered = sorted(values)
    return ordered[min(int(len(ordered) * fraction), len(ordered) - 1)]


def summarise_latency(samples: Sequence[float], *, digits: int = 3) -> dict[str, float]:
    """``{"n", "p50", "p95", "mean"}`` for a list of latency samples.

    An empty list returns all-zero (``n`` = 0), so a configuration that was
    never timed still renders a row. ``digits`` is the rounding applied to the
    three float fields: ``benchmark_ann_index`` reports sub-millisecond ANN
    latencies and wants 3; ``tune_reranking`` reports whole-millisecond rerank
    latencies and passes ``digits=1``.
    """
    if not samples:
        return {"n": 0, "p50": 0.0, "p95": 0.0, "mean": 0.0}
    ordered = sorted(samples)
    return {
        "n": len(ordered),
        "p50": round(percentile(ordered, 0.50), digits),
        "p95": round(percentile(ordered, 0.95), digits),
        "mean": round(statistics.fmean(ordered), digits),
    }
