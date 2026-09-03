#!/usr/bin/env python3
"""One half of the [M4-09] restart proof, run as its own OS process.

``tests/integration/test_human_checkpoint.py`` invokes this twice with
``subprocess.run``: once to drive a claim to the human checkpoint, and once --
in a *different* interpreter, sharing nothing but the database -- to resume it
with a decision. Two ``PostgresSaver`` instances in one process would prove
nothing about surviving a restart; two processes do.

Each subcommand prints one JSON object on stdout, which is the test's assertion
surface.

    python .../checkpoint_restart_worker.py start  --thread T --claim C
    python .../checkpoint_restart_worker.py resume --thread T --decision approve

Not named ``test_*`` on purpose: pytest must not collect it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from langgraph.types import Command  # noqa: E402

from infrastructure.database import create_session_factory  # noqa: E402
from infrastructure.graph.build import build_claim_graph  # noqa: E402
from infrastructure.graph.checkpointer import open_claim_checkpointer  # noqa: E402
from tests.integration._checkpoint_fakes import build_fake_context  # noqa: E402

NARRATIVE = "Bati o carro em uma colisão na avenida no dia 05/01/2026."


def _run(database_url: str, thread_id: str, payload: Any, claim_id: str) -> Any:
    """Compile the graph on a fresh Postgres checkpointer and run one step."""
    session_factory = create_session_factory()
    with open_claim_checkpointer(database_url) as checkpointer:
        graph = build_claim_graph().compile(checkpointer=checkpointer)
        return graph.invoke(
            payload,
            config={"configurable": {"thread_id": thread_id}},
            context=build_fake_context(session_factory),
        )


def _start(args: argparse.Namespace) -> dict[str, Any]:
    out = _run(
        args.database_url,
        args.thread,
        {"claim_id": args.claim, "raw_claim_text": NARRATIVE},
        args.claim,
    )
    interrupts = out.get("__interrupt__") or []
    return {
        "interrupted": bool(interrupts),
        "interrupt_value": interrupts[0].value if interrupts else None,
        "audit_nodes": [event.node for event in out.get("audit_trail", [])],
    }


def _resume(args: argparse.Namespace) -> dict[str, Any]:
    decision: dict[str, Any] = {"decision": args.decision, "notes": args.notes}
    out = _run(args.database_url, args.thread, Command(resume=decision), args.claim)
    recommendation = out.get("recommendation")
    human_decision = out.get("human_decision")
    return {
        "interrupted": "__interrupt__" in out,
        # Dumped rather than repr'd: the test asserts on values, and this also
        # proves the state came back as the real models -- `model_dump` on a
        # dict silently degraded by the serializer would raise here.
        "recommendation": (
            recommendation.model_dump(mode="json") if recommendation else None
        ),
        "human_decision": (
            human_decision.model_dump(mode="json") if human_decision else None
        ),
        "audit_nodes": [event.node for event in out.get("audit_trail", [])],
    }


def main() -> None:
    """Parse the subcommand, run it, print its JSON result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", required=True)
    parser.add_argument("--thread", required=True)
    parser.add_argument("--claim", default="claim-restart-1")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("start")

    resume_parser = subparsers.add_parser("resume")
    resume_parser.add_argument(
        "--decision", choices=("approve", "edit", "reject"), default="approve"
    )
    resume_parser.add_argument("--notes", default="")

    args = parser.parse_args()
    result = _start(args) if args.command == "start" else _resume(args)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
