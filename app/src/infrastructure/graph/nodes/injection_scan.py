"""The injection-scan node: an optional, advisory classifier pass -- [M5-08 Appendix].

Scores every untrusted span already in state -- the claim narrative and each
retrieved clause's excerpt -- with ``runtime.context.classifier`` (a
``infrastructure.graph.context.InjectionClassifierPort``). A flagged span
becomes one ``AuditEvent``; nothing else. The node returns
``{"audit_trail": [...]}`` or ``{}`` and **touches no other state key** -- no
verdict, no citation, no routing decision is ever conditioned on this node's
output, which is what makes "advisory, never blocking" a structural fact
about the code rather than a policy someone could quietly violate later.

When the classifier is off (``runtime.context.classifier`` is
``infrastructure.graph.context.NO_CLASSIFIER``, the default), every
``classify`` call is a trivial no-op -- the node still runs (it is a fixed
member of the parallel fan-out, see ``infrastructure.graph.build``) but costs
nothing worth special-casing.

The deterministic containment this project actually relies on -- delimiters,
schema rejection, metadata-only document trust -- lives in the prompt
builders and the compatibility node ([M5-08]'s main DoD,
``docs/ARCHITECTURE.md``). This node is the M5-08 issue's Appendix: an
external classifier evaluated as an optional defense-in-depth signal, not a
second decision-maker.
"""

from langgraph.runtime import Runtime

from infrastructure.graph.context import GraphContext
from infrastructure.graph.state import AuditEvent, ClaimState

_NODE_INPUT_PREVIEW_CHARS = 200


def injection_scan(
    state: ClaimState, runtime: Runtime[GraphContext]
) -> dict[str, object]:
    """Score the narrative and every clause excerpt; flag, never block."""
    classifier = runtime.context.classifier
    spans: list[tuple[str, str]] = [("claim_narrative", state["raw_claim_text"])]
    spans += [
        (citation.clause_id, citation.excerpt)
        for citation in state.get("citations") or []
    ]

    events: list[AuditEvent] = []
    for source, text in spans:
        result = classifier.classify(text, source=source)
        if result.flagged:
            events.append(
                AuditEvent(
                    node="injection_scan",
                    action="flagged",
                    confidence=result.score,
                    node_input=(
                        f"source={source} label={result.label}"[
                            :_NODE_INPUT_PREVIEW_CHARS
                        ]
                    ),
                )
            )
    return {"audit_trail": events} if events else {}
