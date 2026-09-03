"""The faithfulness / context-relevance LLM judge [M4-10].

The DoD asks for "faithfulness and context-relevance measurement (RAGAS or an
LLM judge), with the judge prompt committed". This module is that judge, and
the prompt is the committed artefact: two ``build_*_prompt`` functions whose
returned text is the whole instruction the judge ever sees. Nothing here is
assembled at run time from a config file, so the prompt in the repository is
the prompt that produced the published number.

**What each metric asks.**

- *Faithfulness* — for one assertion the compatibility node made, do the
  clauses it cited actually say that? The judge sees the assertion and the
  excerpt text of exactly the clauses that assertion cited, and nothing else.
  It is not asked whether the assertion is *right*, only whether the cited
  text supports it -- the distinction between a grounded answer and a correct
  one, which the verdict accuracy measures separately.
- *Context relevance* — of the clauses retrieval returned for this claim, how
  many bear on it at all? Retrieval always returns ``RETRIEVAL_K`` results
  whether or not ten clauses are relevant, so this is the number that says how
  much of the assessment's context was noise.

**Why a different model family.** The assertions come from the reasoning model
and the justification from the fast model -- both DeepSeek. A judge from the
same family grading its own family's output is self-evaluation, the problem
``docs/EVALUATION.md`` already discloses for the golden set's second reviewer.
The judge here is pinned to the same independent model that pass used
(``scripts/review_golden_set_sample.py``): Gemini, single provider, no
fallback. This narrows the self-grading problem; it does not remove it, and
the doc says so.

**Why three passes.** One LLM judgment is a sample, not a measurement. Every
item is judged ``JUDGE_PASSES`` times and the published value is the majority,
carried with the unanimity rate and the spread across passes -- so a reader can
see how much of the number is signal. Raising the pass count is the honest
answer to "how do you know the judge is stable", and it is what
:class:`JudgeAggregate` reports.
"""

import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from application.use_cases.llm_retry_defaults import (
    DEFAULT_LLM_RETRY_DELAY_SECONDS,
    DEFAULT_LLM_RETRY_MAX_ATTEMPTS,
)
from infrastructure.config.llm_client_factory import build_chat_model
from infrastructure.config.settings import LlmSettings

# Pinned exactly as scripts/review_golden_set_sample.py pins its reviewer: one
# provider, no fallback. A judgment has no sane cross-provider substitute, and
# a silent failover would make the published number's provenance a guess.
JUDGE_MODEL = "google/gemini-3.7-flash"
JUDGE_PROVIDER_ORDER = ["google-vertex/global"]

# Passes per item. Odd, so a majority always exists.
JUDGE_PASSES = 3

SupportLevel = Literal["supported", "partially_supported", "unsupported"]

_EXCERPT_CHARS = 1200


class AssertionJudgment(BaseModel):
    """The judge's verdict on one assertion against the clauses it cited."""

    index: int = Field(
        ...,
        description=(
            "O número da afirmação avaliada, exatamente como numerada no prompt."
        ),
    )
    support: SupportLevel = Field(
        ...,
        description=(
            "'supported' se o texto das cláusulas citadas afirma o que a afirmação "
            "diz; 'partially_supported' se sustenta parte dela e o resto não aparece "
            "no texto; 'unsupported' se o texto citado não sustenta a afirmação ou a "
            "contradiz. Julgue APENAS contra o texto mostrado, nunca contra o seu "
            "conhecimento de seguros."
        ),
    )
    rationale: str = Field(
        ..., description="Uma frase justificando a escolha, citando o trecho decisivo."
    )


class FaithfulnessBatch(BaseModel):
    """One judgment per assertion, in the order the prompt numbered them."""

    judgments: list[AssertionJudgment]


class ClauseRelevanceJudgment(BaseModel):
    """The judge's verdict on whether one retrieved clause bears on the claim."""

    clause_id: str = Field(..., description="O clause_id exato mostrado no prompt.")
    relevant: bool = Field(
        ...,
        description=(
            "true se a cláusula tem relação com o evento narrado -- se um analista "
            "de sinistros a leria para decidir este caso, mesmo que a resposta final "
            "seja negativa. false se trata de assunto alheio ao evento."
        ),
    )
    rationale: str = Field(..., description="Uma frase justificando a escolha.")


class ContextRelevanceBatch(BaseModel):
    """One judgment per retrieved clause, in the order the prompt listed them."""

    judgments: list[ClauseRelevanceJudgment]


# --- The committed prompts -------------------------------------------------


_FAITHFULNESS_INSTRUCTIONS = """\
Você é um auditor independente de fundamentação textual. Não é um analista de \
seguros e não deve decidir se o sinistro procede.

Sua única tarefa: para cada afirmação numerada abaixo, dizer se o TEXTO DAS \
CLÁUSULAS CITADAS por aquela afirmação sustenta o que ela diz.

Regras, nesta ordem de precedência:
1. Julgue exclusivamente contra o texto das cláusulas mostrado. Conhecimento \
externo sobre seguros, sobre a lei brasileira ou sobre o que "normalmente" uma \
apólice diz é irrelevante e não pode sustentar uma afirmação.
2. Uma afirmação está 'supported' apenas se o texto citado afirma o seu \
conteúdo. Uma paráfrase fiel conta; uma inferência que exige um passo não \
escrito no texto, não.
3. 'partially_supported' quando o texto sustenta parte da afirmação e o \
restante simplesmente não aparece nele.
4. 'unsupported' quando o texto citado não sustenta a afirmação, trata de \
outro assunto, ou a contradiz.
5. Se a afirmação estiver correta sobre seguros mas o texto citado não a \
disser, ela é 'unsupported'. Estar certo não é o mesmo que estar fundamentado, \
e é a fundamentação que se mede aqui.

Retorne exatamente uma avaliação por afirmação, na mesma ordem e com o mesmo \
número."""


_CONTEXT_RELEVANCE_INSTRUCTIONS = """\
Você é um auditor independente de relevância de contexto. Não decide se o \
sinistro procede.

Sua única tarefa: para cada cláusula recuperada abaixo, dizer se ela tem \
relação com o evento narrado pelo segurado.

Regras:
1. 'relevant' = um analista de sinistros leria essa cláusula para decidir este \
caso. Uma cláusula que EXCLUI o evento é relevante -- decidir contra o segurado \
também é decidir.
2. 'não relevante' = a cláusula trata de assunto alheio ao evento narrado \
(outra cobertura, outro tipo de bem, matéria puramente administrativa sem \
relação com o que aconteceu).
3. Julgue cada cláusula isoladamente, pelo texto mostrado. Não penalize uma \
cláusula por ser redundante com outra da lista.
4. Uma definição de glossário só é relevante se define um termo que o evento \
narrado usa.

Retorne exatamente uma avaliação por cláusula, na mesma ordem, repetindo o \
clause_id exatamente como mostrado."""


def build_faithfulness_prompt(
    claim_text: str,
    assertions: Sequence[tuple[str, Sequence[str]]],
    excerpts: dict[str, str],
) -> str:
    """The committed faithfulness prompt for one claim's assertions.

    ``assertions`` is ``(statement, clause_ids)`` as
    [infrastructure.graph.nodes.compatibility.parse_reasoning] recovers it;
    ``excerpts`` maps clause id to the excerpt text retrieval actually showed
    the model -- the judge is held to the same evidence the assessor had, not
    to the full corpus clause.
    """
    blocks: list[str] = []
    for index, (statement, clause_ids) in enumerate(assertions, start=1):
        cited = "\n".join(
            f"    [{clause_id}] "
            f"{excerpts.get(clause_id, '(texto indisponível)')[:_EXCERPT_CHARS]}"
            for clause_id in clause_ids
        )
        blocks.append(
            f"Afirmação {index}: {statement}\n"
            f"  Cláusulas citadas por esta afirmação:\n{cited or '    (nenhuma)'}"
        )
    return (
        f"{_FAITHFULNESS_INSTRUCTIONS}\n\n"
        f"=== Relato do segurado ===\n{claim_text}\n\n"
        f"=== Afirmações a avaliar ({len(assertions)}) ===\n"
        + "\n\n".join(blocks)
        + f"\n\nRetorne exatamente {len(assertions)} avaliação(ões)."
    )


def build_context_relevance_prompt(
    claim_text: str, clauses: Sequence[tuple[str, str]]
) -> str:
    """The committed context-relevance prompt for one claim's retrieved clauses.

    ``clauses`` is ``(clause_id, excerpt)`` in the order retrieval ranked them.
    """
    listed = "\n\n".join(
        f"[{clause_id}]\n{excerpt[:_EXCERPT_CHARS]}" for clause_id, excerpt in clauses
    )
    return (
        f"{_CONTEXT_RELEVANCE_INSTRUCTIONS}\n\n"
        f"=== Relato do segurado ===\n{claim_text}\n\n"
        f"=== Cláusulas recuperadas ({len(clauses)}) ===\n{listed}"
        f"\n\nRetorne exatamente {len(clauses)} avaliação(ões)."
    )


# --- Running the judge -----------------------------------------------------


@dataclass(frozen=True)
class JudgeAggregate:
    """One item judged ``n_passes`` times: the majority plus how stable it was.

    ``unanimous`` is True when every pass agreed. ``pass_values`` keeps each
    pass's raw label so a report can show the disagreement rather than assert
    stability.
    """

    majority: str
    pass_values: tuple[str, ...]

    @property
    def unanimous(self) -> bool:
        """True when every pass returned the same label."""
        return len(set(self.pass_values)) == 1

    @property
    def n_passes(self) -> int:
        """How many passes produced this aggregate."""
        return len(self.pass_values)


def majority_of(values: Sequence[str]) -> JudgeAggregate:
    """Majority label across passes; ties break toward the first pass's answer.

    A tie is only reachable with an even pass count, which ``JUDGE_PASSES``
    is not -- the rule exists so a caller overriding the count still gets a
    defined, order-stable answer rather than an arbitrary one.
    """
    if not values:
        raise ValueError("majority_of needs at least one pass")
    counts = Counter(values)
    best = max(counts.values())
    for value in values:  # first pass order, so ties are deterministic
        if counts[value] == best:
            return JudgeAggregate(majority=value, pass_values=tuple(values))
    raise AssertionError("unreachable")  # pragma: no cover


def build_judge_model(settings: LlmSettings) -> BaseChatModel:
    """The pinned judge model: independent family, one provider, no fallback."""
    return build_chat_model(
        settings,
        JUDGE_MODEL,
        provider_order=JUDGE_PROVIDER_ORDER,
        allow_fallbacks=False,
    )


def invoke_judge[T: BaseModel](
    model: BaseChatModel,
    prompt: str,
    schema: type[T],
    *,
    max_attempts: int = DEFAULT_LLM_RETRY_MAX_ATTEMPTS,
    delay_seconds: float = DEFAULT_LLM_RETRY_DELAY_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Invoke the judge with structured output, retrying transient failures.

    Re-raises after ``max_attempts``: there is no sane fallback value for a
    judgment, and a silently defaulted one would enter a published average.
    """
    chain = cast(Runnable[str, T], model.with_structured_output(schema))
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return chain.invoke(prompt)
        except Exception as exc:  # noqa: BLE001 - retried below, re-raised at the end
            last_exc = exc
            if attempt < max_attempts:
                sleep(delay_seconds)
    assert last_exc is not None
    raise last_exc


def judge_faithfulness(
    model: BaseChatModel,
    claim_text: str,
    assertions: Sequence[tuple[str, Sequence[str]]],
    excerpts: dict[str, str],
    *,
    passes: int = JUDGE_PASSES,
) -> list[JudgeAggregate]:
    """Judge every assertion ``passes`` times; one aggregate per assertion.

    A pass that returns the wrong number of judgments, or numbers them
    inconsistently, is padded with ``"unsupported"`` for the assertions it
    skipped -- the conservative direction, and visible in ``pass_values`` as a
    disagreement rather than hidden as a silently dropped item.
    """
    if not assertions:
        return []
    prompt = build_faithfulness_prompt(claim_text, assertions, excerpts)
    per_assertion: list[list[str]] = [[] for _ in assertions]
    for _ in range(passes):
        batch = invoke_judge(model, prompt, FaithfulnessBatch)
        by_index = {j.index: j.support for j in batch.judgments}
        for position in range(len(assertions)):
            per_assertion[position].append(by_index.get(position + 1, "unsupported"))
    return [majority_of(values) for values in per_assertion]


def judge_context_relevance(
    model: BaseChatModel,
    claim_text: str,
    clauses: Sequence[tuple[str, str]],
    *,
    passes: int = JUDGE_PASSES,
) -> dict[str, JudgeAggregate]:
    """Judge every retrieved clause ``passes`` times; one aggregate per clause id.

    Values are the strings ``"relevant"`` / ``"irrelevant"``, so the same
    :class:`JudgeAggregate` machinery carries both metrics. A clause a pass
    failed to judge counts as ``"irrelevant"`` for that pass, the conservative
    direction for a *relevance* rate.
    """
    if not clauses:
        return {}
    prompt = build_context_relevance_prompt(claim_text, clauses)
    per_clause: dict[str, list[str]] = {clause_id: [] for clause_id, _ in clauses}
    for _ in range(passes):
        batch = invoke_judge(model, prompt, ContextRelevanceBatch)
        by_id = {j.clause_id: j.relevant for j in batch.judgments}
        for clause_id in per_clause:
            per_clause[clause_id].append(
                "relevant" if by_id.get(clause_id, False) else "irrelevant"
            )
    return {clause_id: majority_of(values) for clause_id, values in per_clause.items()}
