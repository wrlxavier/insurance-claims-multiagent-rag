"""Exclusion co-retrieval: pair a coverage clause with its exclusion -- [M3-06].

The core retrieval problem of the domain: a coverage clause retrieved without
the exclusion that limits it produces an assessment that is fluent, well-cited
and wrong. The golden set's ``coverage_with_exclusion`` questions reference
*both* clauses on purpose; lexical and dense retrieval bring back the coverage
clause and miss the exclusion a few paragraphs away (``coverage_with_exclusion``
is the weakest question type -- see ``docs/RERANKING.md``).

[ExclusionCoRetrievalRetriever] wraps any filtered retriever (in practice
[infrastructure.rag.reranking_retriever.RerankingRetriever]) behind the **same**
``retrieve(question, *, k, metadata_filter)`` interface, so the [M2-06] harness
and M4's graph node stay unaware of it -- exactly the composition
``RerankingRetriever`` uses over ``HybridRetriever``. After the base ranks, for
every retrieved ``coverage`` clause it pulls the linked ``exclusion`` clauses
from [ClauseGraph] and reserves
[infrastructure.rag.exclusion_co_retrieval_config.RESERVED_EXCLUSION_SLOTS] of
the final top-k for them.

[ClauseGraph] is a pure, deterministic structural index over the parsed corpus
([infrastructure.parsing.clause_schema.ParsedClauseRecord]) -- no DB, no LLM. It
links a coverage clause to exclusion clauses by three edges ([M3-06] DoD):

- ``same_section`` -- the exclusion shares the coverage clause's top-level
  section root (siblings, descendants of the coverage clause, exclusions
  elsewhere in the same numbered section).
- ``adjacent_section`` -- the exclusion is in a *different* section root whose
  page span is within
  [config.ADJACENT_SECTION_MAX_PAGE_GAP] pages (the flat / parent-less OCR
  documents, where ``same_section`` cannot fire because every clause is its own
  root).
- ``cross_reference`` -- a ``cláusula N.M`` token in the coverage clause's text
  resolves to an exclusion clause by numbering.

Deviation from the DoD, recorded per the repo convention ([M3-04]
RRF-vs-weighted, [M1-08c] predictions): the DoD says parse cross-references
"during M1". M1's corpus schema is frozen and a re-parse (OCR + LLM
classification + vision escalation) is disproportionate, so cross-reference
edges are derived deterministically from the already-persisted ``clause.text``
here, at graph-build time -- the same thing ``scripts/find_candidate_clauses.py``
already does for [M2-08] curation.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import IntEnum
from typing import Protocol

from domain.clause_classification import ClauseType
from infrastructure.parsing.clause_schema import ParsedClauseRecord
from infrastructure.rag.exclusion_co_retrieval_config import (
    ADJACENT_SECTION_MAX_PAGE_GAP,
    CROSS_REFERENCE_PATTERN,
    RESERVED_EXCLUSION_SLOTS,
)
from infrastructure.rag.retrieval_filter import RetrievalFilter

_CROSS_REFERENCE_RE = re.compile(CROSS_REFERENCE_PATTERN, re.IGNORECASE)


def extract_cross_references(text: str) -> set[str]:
    """Return the numbering tokens textually referenced in ``text``.

    Matches "cláusula 12", "Cláusula 4.2", "conforme cláusula 10.2" and returns
    the bare numbering ("12", "4.2", "10.2"). The production formalisation of
    ``scripts/find_candidate_clauses.py``'s reduced-scope helper of the same name
    ([M2-08]); the curation script keeps its own copy so the four frozen M2
    authoring scripts that import it are untouched.
    """
    return {match.group(1) for match in _CROSS_REFERENCE_RE.finditer(text)}


class ExclusionEdge(IntEnum):
    """How an exclusion clause is linked to a coverage clause, best first.

    The integer order is the ranking priority: a cross-referenced exclusion
    outranks a same-section one, which outranks an adjacent-section one.
    """

    CROSS_REFERENCE = 0
    SAME_SECTION = 1
    ADJACENT_SECTION = 2


@dataclass(frozen=True, order=True)
class LinkedExclusion:
    """One exclusion clause linked to a retrieved coverage clause.

    Field order is the sort key: edge kind, then tree distance (only meaningful
    within a section, otherwise 0), then page gap, then clause id -- fully
    deterministic, with no reliance on dict order.
    """

    edge: ExclusionEdge
    tree_distance: int
    page_gap: int
    clause_id: str


class ClauseGraph:
    """Deterministic structural index linking coverage clauses to exclusions."""

    def __init__(self, records: Iterable[ParsedClauseRecord]) -> None:
        """Build the index from the parsed corpus (one pass, then memoised walks)."""
        self._by_id: dict[str, ParsedClauseRecord] = {
            record.clause_id: record for record in records
        }
        self._exclusion_ids_by_document: dict[str, list[str]] = {}
        # (document_id, last path segment) -> clause ids carrying that numbering.
        self._by_numbering: dict[tuple[str, str], list[str]] = {}
        self._section_root_cache: dict[str, str] = {}
        self._ancestors_cache: dict[str, tuple[str, ...]] = {}
        for record in self._by_id.values():
            if record.clause_type is ClauseType.EXCLUSION:
                self._exclusion_ids_by_document.setdefault(
                    record.document_id, []
                ).append(record.clause_id)
            numbering = record.path.rsplit("/", 1)[-1]
            self._by_numbering.setdefault((record.document_id, numbering), []).append(
                record.clause_id
            )

    def is_coverage(self, clause_id: str) -> bool:
        """Whether ``clause_id`` is a known ``coverage`` clause."""
        record = self._by_id.get(clause_id)
        return record is not None and record.clause_type is ClauseType.COVERAGE

    def is_exclusion(self, clause_id: str) -> bool:
        """Whether ``clause_id`` is a known ``exclusion`` clause."""
        record = self._by_id.get(clause_id)
        return record is not None and record.clause_type is ClauseType.EXCLUSION

    def _ancestors(self, clause_id: str) -> tuple[str, ...]:
        """``clause_id`` then each parent up to the section root, root last.

        A ``parent_id`` pointing outside the corpus, or a cycle, stops the walk
        -- the last id reached is treated as the root.
        """
        cached = self._ancestors_cache.get(clause_id)
        if cached is not None:
            return cached
        chain: list[str] = []
        seen: set[str] = set()
        current: str | None = clause_id
        while current is not None and current not in seen and current in self._by_id:
            chain.append(current)
            seen.add(current)
            current = self._by_id[current].parent_id
        result = tuple(chain)
        self._ancestors_cache[clause_id] = result
        return result

    def _section_root(self, clause_id: str) -> str:
        """The top-level ancestor of ``clause_id`` (itself when it has no parent)."""
        cached = self._section_root_cache.get(clause_id)
        if cached is not None:
            return cached
        ancestors = self._ancestors(clause_id)
        root = ancestors[-1] if ancestors else clause_id
        self._section_root_cache[clause_id] = root
        return root

    def _tree_distance(self, a_id: str, b_id: str) -> int:
        """Edge count between two clauses in the tree, or 0 if unrelated."""
        a_chain = self._ancestors(a_id)
        b_index = {clause_id: i for i, clause_id in enumerate(self._ancestors(b_id))}
        for i, clause_id in enumerate(a_chain):
            if clause_id in b_index:
                return i + b_index[clause_id]
        return 0

    @staticmethod
    def _page_gap(a: ParsedClauseRecord, b: ParsedClauseRecord) -> int:
        """Pages between two clause spans; 0 when the spans touch or overlap."""
        return max(0, max(a.page_start, b.page_start) - min(a.page_end, b.page_end))

    def _bundle_compatible(
        self, coverage: ParsedClauseRecord, candidate: ParsedClauseRecord
    ) -> bool:
        """A bundle-doc guard: keep same-bundle or unknown-bundle exclusions only."""
        if coverage.bundle_section is None:
            return True
        return candidate.bundle_section in (coverage.bundle_section, None)

    def linked_exclusions(self, coverage_clause_id: str) -> list[LinkedExclusion]:
        """Every exclusion clause linked to ``coverage_clause_id``, ranked best first.

        Empty when ``coverage_clause_id`` is unknown or not a coverage clause. A
        clause reachable by more than one edge is kept once, under its best
        (lowest-sorting) link.
        """
        coverage = self._by_id.get(coverage_clause_id)
        if coverage is None or coverage.clause_type is not ClauseType.COVERAGE:
            return []

        best: dict[str, LinkedExclusion] = {}

        def offer(link: LinkedExclusion) -> None:
            current = best.get(link.clause_id)
            if current is None or link < current:
                best[link.clause_id] = link

        coverage_root = self._section_root(coverage_clause_id)

        for exclusion_id in self._exclusion_ids_by_document.get(
            coverage.document_id, []
        ):
            if exclusion_id == coverage_clause_id:
                continue
            candidate = self._by_id[exclusion_id]
            if not self._bundle_compatible(coverage, candidate):
                continue
            if self._section_root(exclusion_id) == coverage_root:
                offer(
                    LinkedExclusion(
                        edge=ExclusionEdge.SAME_SECTION,
                        tree_distance=self._tree_distance(
                            coverage_clause_id, exclusion_id
                        ),
                        page_gap=self._page_gap(coverage, candidate),
                        clause_id=exclusion_id,
                    )
                )
            else:
                gap = self._page_gap(coverage, candidate)
                if gap <= ADJACENT_SECTION_MAX_PAGE_GAP:
                    offer(
                        LinkedExclusion(
                            edge=ExclusionEdge.ADJACENT_SECTION,
                            tree_distance=0,
                            page_gap=gap,
                            clause_id=exclusion_id,
                        )
                    )

        for numbering in extract_cross_references(coverage.text):
            for candidate_id in self._by_numbering.get(
                (coverage.document_id, numbering), []
            ):
                if candidate_id == coverage_clause_id or not self.is_exclusion(
                    candidate_id
                ):
                    continue
                candidate = self._by_id[candidate_id]
                if not self._bundle_compatible(coverage, candidate):
                    continue
                offer(
                    LinkedExclusion(
                        edge=ExclusionEdge.CROSS_REFERENCE,
                        tree_distance=0,
                        page_gap=self._page_gap(coverage, candidate),
                        clause_id=candidate_id,
                    )
                )

        return sorted(best.values())

    def ranked_linked_exclusions(
        self, coverage_clause_ids: Sequence[str]
    ) -> list[LinkedExclusion]:
        """Merge :meth:`linked_exclusions` over several coverage clauses, ranked.

        An exclusion linked to more than one retrieved coverage clause is kept
        once, under its best link.
        """
        best: dict[str, LinkedExclusion] = {}
        for coverage_clause_id in coverage_clause_ids:
            for link in self.linked_exclusions(coverage_clause_id):
                current = best.get(link.clause_id)
                if current is None or link < current:
                    best[link.clause_id] = link
        return sorted(best.values())


class _FilterableBase(Protocol):
    """The wrapped retriever's contract: a filtered, clause-id-ranking ``retrieve``.

    Re-declared here (rather than imported from ``infrastructure.evaluation``) so
    ``infrastructure.rag`` still imports nothing from that package -- the same
    split ``reranking_retriever._FilterableBase`` keeps.
    """

    def retrieve(
        self,
        question: str,
        *,
        k: int,
        metadata_filter: RetrievalFilter | None = None,
    ) -> list[str]:
        """Up to ``k`` clause ids, ranked best-match first, filtered."""
        ...


class ExclusionCoRetrievalRetriever:
    """Reserves top-k budget for the exclusions that limit retrieved coverage."""

    def __init__(
        self,
        base: _FilterableBase,
        graph: ClauseGraph,
        *,
        reserved_slots: int = RESERVED_EXCLUSION_SLOTS,
    ) -> None:
        """Compose the base retriever with the structural clause graph."""
        self._base = base
        self._graph = graph
        self._reserved_slots = reserved_slots

    def retrieve(
        self,
        question: str,
        *,
        k: int,
        metadata_filter: RetrievalFilter | None = None,
    ) -> list[str]:
        """Up to ``k`` clause ids: the base ranking with linked exclusions kept.

        For the retrieved coverage clauses, the ``reserved_slots`` best-linked
        exclusions the base ranked outside the top-k are each given a slot. The
        output is identical to the base ranking when nothing links, or when every
        linked exclusion is already in the top-k.
        """
        if k <= 0:
            return []
        ranked = self._base.retrieve(question, k=k, metadata_filter=metadata_filter)
        reserved = min(self._reserved_slots, k)
        if reserved <= 0:
            return ranked

        coverage_ids = [cid for cid in ranked if self._graph.is_coverage(cid)]
        if not coverage_ids:
            return ranked

        links = self._graph.ranked_linked_exclusions(coverage_ids)
        if not links:
            return ranked

        ranked_set = set(ranked)
        to_inject = [
            link.clause_id for link in links if link.clause_id not in ranked_set
        ][:reserved]
        if not to_inject:
            return ranked

        return self._reserve(ranked, to_inject, k)

    def _reserve(
        self, ranked: Sequence[str], to_inject: Sequence[str], k: int
    ) -> list[str]:
        """Append ``to_inject`` after ``ranked``, evicting supporting entries.

        To hold the length at ``k``, one lowest-ranked base entry is dropped per
        injected clause -- but only entries that are neither ``coverage`` nor
        ``exclusion``. A coverage clause is the primary answer and an exclusion
        the base already surfaced is the whole point of this step, so if nothing
        else is left to drop the injection stops rather than displacing one.
        """
        survivors = list(ranked)
        injected: list[str] = []
        for clause_id in to_inject:
            if len(survivors) + len(injected) < k:
                injected.append(clause_id)
                continue
            drop_index = next(
                (
                    i
                    for i in range(len(survivors) - 1, -1, -1)
                    if not self._graph.is_coverage(survivors[i])
                    and not self._graph.is_exclusion(survivors[i])
                ),
                None,
            )
            if drop_index is None:
                break
            survivors.pop(drop_index)
            injected.append(clause_id)
        return survivors + injected
