"""Deterministic stand-in for the LLM classification pass.

``scripts/build_corpus.py`` runs with this classifier by default: no
[langchain_core.language_models.chat_models.BaseChatModel] factory exists
in this repo yet (see [infrastructure.parsing.llm_classifier
.LangchainClauseClassifier], which requires one already built), and
calling a real LLM would make ``make parse`` depend on an API key and
network access, and its output non-deterministic run to run -- directly at
odds with [M1-07]'s "reproducible build" goal. Every clause the rule pass
in [application.use_cases.clause_classification.classify_and_enrich_clauses]
leaves unmatched falls through to this classifier, which always raises;
that function's existing ``except Exception`` fallback then assigns
``ClauseType.OTHER``/``TypeSource.LLM``/``confidence=0.0`` deterministically.
Wiring a real LLM here is left to a future issue.
"""

from domain.clause_classification import ClauseType


class NullClauseClassifier:
    """A [application.ports.clause_classifier.ClauseClassifierPort] stub."""

    def classify(self, clause_title: str, clause_text: str) -> tuple[ClauseType, float]:
        """Always raise; the caller's OTHER/0.0 fallback then applies."""
        raise NotImplementedError(
            "No LLM classifier is wired up yet -- see this module's docstring."
        )
