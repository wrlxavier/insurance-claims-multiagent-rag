"""Run-scoped dependencies injected into every graph node -- [M4-01b].

``GraphContext`` is built once outside this package (the composition root, a
later issue) and handed to the graph as
``graph.invoke(state, context=GraphContext(...))``. Each node receives it as
``runtime.context`` by declaring a ``runtime: Runtime[GraphContext]`` parameter.
This is the only channel for the chat models, the retrieval port and the model
config -- a node holds no module-level client and takes no constructor. See
``docs/ARCHITECTURE.md`` ([M4-01b]).
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from langchain_core.language_models.chat_models import BaseChatModel

from infrastructure.config.settings import LlmSettings

if TYPE_CHECKING:
    from infrastructure.rag.retrieval_filter import RetrievalFilter


class RetrievalPort(Protocol):
    """The retrieval capability a node depends on, owned by the graph layer.

    The signature mirrors the M3 retrievers
    (``infrastructure.evaluation.retriever.FilterableRetriever``) so a concrete
    hybrid/rerank retriever satisfies it structurally, without the graph
    importing a concrete implementation or the evaluation harness's interface
    ([M4-04]). ``retrieve`` returns ranked clause-id strings only -- the
    retrieval node hydrates ``Citation`` objects itself. [M4-04] owns any
    change to this shape.
    """

    def retrieve(
        self,
        question: str,
        *,
        k: int,
        metadata_filter: "RetrievalFilter | None" = None,
    ) -> list[str]:
        """Return up to ``k`` clause ids, ranked best-match first."""
        ...


@dataclass(frozen=True)
class GraphContext:
    """Run-scoped dependencies for the agent graph -- [M4-01b].

    Frozen: a node reads these, it does not swap them. The two chat models are
    pre-built so a unit test passes fakes directly. ``llm_settings`` is the
    model config -- a node records ``llm_settings.llm_model_reasoning`` (or
    ``.llm_model_fast``) as the ``AuditEvent.model`` for a call it makes,
    rather than introspecting the model object. The OpenRouter provider pins
    are applied by ``build_chat_model`` at the composition root, never read by
    a node.
    """

    fast_model: BaseChatModel
    reasoning_model: BaseChatModel
    retriever: RetrievalPort
    llm_settings: LlmSettings
