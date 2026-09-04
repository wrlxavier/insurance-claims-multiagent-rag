"""In-memory fakes and builders for the M5-02 application-layer tests.

No mock library, no LLM, no database, no graph. Every use-case test in
``tests/unit/application`` drives the interactors through these.
"""

from collections.abc import Iterable, Sequence
from datetime import UTC, datetime, timedelta

from application.assessment_job import AssessmentJob, JobStatus
from application.assessment_record import AssessmentRecord, AssessmentStatus
from application.audit_trail_entry import AuditTrailEntry
from application.consistency_flag import ConsistencyFlag
from application.orchestrator_result import OrchestratorResult
from application.ports.assessment_job_repository import AssessmentJobRepository
from application.ports.assessment_repository import AssessmentRepository
from application.ports.audit_trail_writer import AuditTrailWriter
from application.ports.unit_of_work import UnitOfWork, UnitOfWorkFactory
from domain.citation import Citation
from domain.claim import Claim
from domain.clause_classification import ClauseType
from domain.human_decision import HumanDecision
from domain.policy_clause import PolicyClause
from domain.susep_process import SusepProcess
from domain.verdict import Verdict

SUSEP = SusepProcess("15414.610650/2024-59")
FIXED_NOW = datetime(2026, 9, 3, 12, 0, tzinfo=UTC)


# --------------------------------------------------------------------------- #
# builders
# --------------------------------------------------------------------------- #
def make_citation(
    clause_id: str = "15414610650202459:2.1", **overrides: object
) -> Citation:
    fields: dict[str, object] = {
        "clause_id": clause_id,
        "document_id": "15414610650202459",
        "susep_process": SUSEP,
        "clause_type": ClauseType.COVERAGE,
        "excerpt": "A cobertura compreende colisao, incendio e roubo.",
        "relevance_score": 0.83,
    }
    fields.update(overrides)
    return Citation(**fields)  # type: ignore[arg-type]


def make_policy_clause(
    clause_id: str = "15414610650202459:2.1", **overrides: object
) -> PolicyClause:
    fields: dict[str, object] = {
        "clause_id": clause_id,
        "susep_process": SUSEP,
        "document_id": "15414610650202459",
        "clause_type": ClauseType.COVERAGE,
        "text": "A cobertura compreende colisao, incendio e roubo.",
        "heading": "2.1 Coberturas",
    }
    fields.update(overrides)
    return PolicyClause(**fields)  # type: ignore[arg-type]


def make_consistency_flag(**overrides: object) -> ConsistencyFlag:
    fields: dict[str, object] = {
        "check": "event_date_within_policy_period",
        "severity": "attention",
        "detail": "A data do evento antecede o inicio de vigencia.",
        "source": "deterministic",
    }
    fields.update(overrides)
    return ConsistencyFlag(**fields)  # type: ignore[arg-type]


def make_orchestrator_result(**overrides: object) -> OrchestratorResult:
    fields: dict[str, object] = {
        "verdict": Verdict.COMPATIBLE,
        "reasoning": "O evento descrito e uma colisao, coberta pela clausula 2.1.",
        "recommended_action": "Encaminhar para analise humana.",
        "citations": (make_citation(),),
        "confidence": 0.72,
        "consistency_flags": (),
        "context_sufficient": True,
        "clarification_exhausted": False,
        "missing_information": (),
        "awaiting_review": True,
    }
    fields.update(overrides)
    return OrchestratorResult(**fields)  # type: ignore[arg-type]


def abstain_result(**overrides: object) -> OrchestratorResult:
    """An insufficient-context run: a recommendation grounded in nothing."""
    fields: dict[str, object] = {
        "verdict": Verdict.INSUFFICIENT_INFORMATION,
        "reasoning": "O contexto recuperado nao permite decidir.",
        "recommended_action": "Solicitar documentacao adicional ao segurado.",
        "citations": (),
        "confidence": 0.2,
        "context_sufficient": False,
        "missing_information": ("data_evento_vigencia",),
    }
    fields.update(overrides)
    return make_orchestrator_result(**fields)


def make_audit_entry(**overrides: object) -> AuditTrailEntry:
    fields: dict[str, object] = {
        "sequence": 0,
        "timestamp": FIXED_NOW,
        "node": "recommendation",
        "action": "consolidate",
        "node_input": "posture=compatible verdict=compatible n_clauses=1",
    }
    fields.update(overrides)
    return AuditTrailEntry(**fields)  # type: ignore[arg-type]


def make_job(**overrides: object) -> AssessmentJob:
    fields: dict[str, object] = {
        "assessment_id": "assessment-1",
        "claim_id": "claim-1",
        "raw_text": "Bati o carro na traseira de outro veiculo.",
        "policy_ref": None,
        "submitted_at": FIXED_NOW,
        "status": JobStatus.PENDING,
        "attempts": 0,
        "created_at": FIXED_NOW,
        "updated_at": FIXED_NOW,
        "failure": None,
    }
    fields.update(overrides)
    return AssessmentJob(**fields)  # type: ignore[arg-type]


def make_record(**overrides: object) -> AssessmentRecord:
    fields: dict[str, object] = {
        "assessment_id": "assessment-1",
        "claim_id": "claim-1",
        "verdict": Verdict.COMPATIBLE,
        "reasoning": "O evento descrito e uma colisao, coberta pela clausula 2.1.",
        "recommended_action": "Encaminhar para analise humana.",
        "citations": (make_citation(),),
        "confidence": 0.72,
        "consistency_flags": (),
        "context_sufficient": True,
        "clarification_exhausted": False,
        "missing_information": (),
        "status": AssessmentStatus.AWAITING_REVIEW,
        "created_at": FIXED_NOW,
        "decision": None,
    }
    fields.update(overrides)
    return AssessmentRecord(**fields)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# fakes
# --------------------------------------------------------------------------- #
class FixedClock:
    """A clock stuck at one instant, nudgeable with ``advance``."""

    def __init__(self, moment: datetime = FIXED_NOW) -> None:
        self._moment = moment

    def now(self) -> datetime:
        return self._moment

    def advance(self, delta: timedelta) -> None:
        self._moment = self._moment + delta

    def set(self, moment: datetime) -> None:
        self._moment = moment


class NaiveClock:
    """A misconfigured clock: returns a naive datetime, which the domain rejects."""

    def now(self) -> datetime:
        return FIXED_NOW.replace(tzinfo=None)


class SequentialIds:
    """A deterministic ``new_id`` -- ``prefix-1``, ``prefix-2``, ..."""

    def __init__(self, prefix: str = "id") -> None:
        self._prefix = prefix
        self._n = 0
        self.issued: list[str] = []

    def __call__(self) -> str:
        self._n += 1
        value = f"{self._prefix}-{self._n}"
        self.issued.append(value)
        return value


class InMemoryClauseRepository:
    """A clause corpus backed by a dict, keyed by ``clause_id``."""

    def __init__(self, clauses: Iterable[PolicyClause] = ()) -> None:
        self._by_id = {clause.clause_id: clause for clause in clauses}

    def get(self, clause_id: str) -> PolicyClause | None:
        return self._by_id.get(clause_id)

    def get_many(self, clause_ids: Sequence[str]) -> tuple[PolicyClause, ...]:
        return tuple(
            self._by_id[clause_id]
            for clause_id in clause_ids
            if clause_id in self._by_id
        )

    def list_for_policy(self, policy: SusepProcess) -> tuple[PolicyClause, ...]:
        return tuple(
            clause for clause in self._by_id.values() if clause.susep_process == policy
        )


class InMemoryAssessmentRepository:
    """An assessment store over a shared dict (so a UoW can snapshot it)."""

    def __init__(self, store: dict[str, AssessmentRecord]) -> None:
        self._store = store

    def add(self, record: AssessmentRecord) -> None:
        if record.assessment_id in self._store:
            raise KeyError(f"assessment {record.assessment_id!r} already exists")
        self._store[record.assessment_id] = record

    def update(self, record: AssessmentRecord) -> None:
        if record.assessment_id not in self._store:
            raise KeyError(f"assessment {record.assessment_id!r} does not exist")
        self._store[record.assessment_id] = record

    def get(self, assessment_id: str) -> AssessmentRecord | None:
        return self._store.get(assessment_id)

    def list(
        self,
        *,
        claim_id: str | None = None,
        status: AssessmentStatus | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[AssessmentRecord, ...]:
        rows = [
            record
            for record in self._store.values()
            if (claim_id is None or record.claim_id == claim_id)
            and (status is None or record.status is status)
        ]
        rows.sort(key=lambda record: record.assessment_id)
        rows.sort(key=lambda record: record.created_at, reverse=True)
        return tuple(rows[offset : offset + limit])


class InMemoryAssessmentJobRepository:
    """A job store over a shared dict (so a UoW can snapshot it)."""

    def __init__(self, store: dict[str, AssessmentJob]) -> None:
        self._store = store

    def add(self, job: AssessmentJob) -> None:
        if job.assessment_id in self._store:
            raise KeyError(f"assessment job {job.assessment_id!r} already exists")
        self._store[job.assessment_id] = job

    def update(self, job: AssessmentJob) -> None:
        if job.assessment_id not in self._store:
            raise KeyError(f"assessment job {job.assessment_id!r} does not exist")
        self._store[job.assessment_id] = job

    def get(self, assessment_id: str) -> AssessmentJob | None:
        return self._store.get(assessment_id)


class FakeAssessmentQueue:
    """Records every ``enqueue`` call instead of touching Redis."""

    def __init__(self) -> None:
        self.enqueued: list[str] = []

    def enqueue(self, assessment_id: str) -> None:
        self.enqueued.append(assessment_id)


AuditStore = dict[str, list[AuditTrailEntry]]
JobStore = dict[str, AssessmentJob]


class InMemoryAuditTrailWriter:
    """Append captured audit entries to a shared store, idempotent on sequence."""

    def __init__(self, store: AuditStore) -> None:
        self._store = store

    def append(
        self,
        *,
        claim_id: str,
        thread_id: str,
        entries: Sequence[AuditTrailEntry],
    ) -> None:
        trail = self._store.setdefault(thread_id, [])
        seen = {entry.sequence for entry in trail}
        for entry in entries:
            if entry.sequence not in seen:
                trail.append(entry)
                seen.add(entry.sequence)
        trail.sort(key=lambda entry: entry.sequence)


class InMemoryAuditTrailReader:
    """Read a thread's captured audit trail from a shared store."""

    def __init__(self, store: AuditStore) -> None:
        self._store = store

    def get_trail(self, assessment_id: str) -> tuple[AuditTrailEntry, ...]:
        return tuple(
            sorted(
                self._store.get(assessment_id, ()),
                key=lambda entry: entry.sequence,
            )
        )


class InMemoryUnitOfWork:
    """A transaction with real rollback: it snapshots every store on entry."""

    assessments: AssessmentRepository
    audit: AuditTrailWriter
    jobs: AssessmentJobRepository

    def __init__(
        self,
        store: dict[str, AssessmentRecord],
        audit_store: AuditStore | None = None,
        job_store: JobStore | None = None,
    ) -> None:
        self._store = store
        self._audit_store: AuditStore = {} if audit_store is None else audit_store
        self._job_store: JobStore = {} if job_store is None else job_store
        self.assessments = InMemoryAssessmentRepository(store)
        self.audit = InMemoryAuditTrailWriter(self._audit_store)
        self.jobs = InMemoryAssessmentJobRepository(self._job_store)
        self._snapshot: dict[str, AssessmentRecord] | None = None
        self._audit_snapshot: AuditStore | None = None
        self._job_snapshot: JobStore | None = None
        self.committed = False

    def __enter__(self) -> "UnitOfWork":
        self._snapshot = dict(self._store)
        self._audit_snapshot = {
            thread: list(trail) for thread, trail in self._audit_store.items()
        }
        self._job_snapshot = dict(self._job_store)
        self.committed = False
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if not self.committed:
            self.rollback()
        self._snapshot = None
        self._audit_snapshot = None
        self._job_snapshot = None

    def commit(self) -> None:
        self.committed = True

    def rollback(self) -> None:
        if self._snapshot is not None:
            self._store.clear()
            self._store.update(self._snapshot)
        if self._audit_snapshot is not None:
            self._audit_store.clear()
            self._audit_store.update(self._audit_snapshot)
        if self._job_snapshot is not None:
            self._job_store.clear()
            self._job_store.update(self._job_snapshot)


def make_uow_factory(
    store: dict[str, AssessmentRecord],
    audit_store: AuditStore | None = None,
    job_store: JobStore | None = None,
) -> UnitOfWorkFactory:
    shared_audit: AuditStore = {} if audit_store is None else audit_store
    shared_jobs: JobStore = {} if job_store is None else job_store

    def _open() -> InMemoryUnitOfWork:
        return InMemoryUnitOfWork(store, shared_audit, shared_jobs)

    return _open


class FakeClaimAssessmentOrchestrator:
    """Records every call; returns canned results or raises a canned error."""

    def __init__(
        self,
        *,
        start_result: OrchestratorResult | None = None,
        resume_result: OrchestratorResult | None = None,
        raise_on_start: Exception | None = None,
        raise_on_resume: Exception | None = None,
    ) -> None:
        self._start_result = start_result or make_orchestrator_result(
            awaiting_review=True
        )
        self._resume_result = resume_result or make_orchestrator_result(
            awaiting_review=False
        )
        self._raise_on_start = raise_on_start
        self._raise_on_resume = raise_on_resume
        self.started: list[tuple[str, Claim]] = []
        self.resumed: list[tuple[str, HumanDecision]] = []

    def start(self, *, assessment_id: str, claim: Claim) -> OrchestratorResult:
        self.started.append((assessment_id, claim))
        if self._raise_on_start is not None:
            raise self._raise_on_start
        return self._start_result

    def resume(
        self, *, assessment_id: str, decision: HumanDecision
    ) -> OrchestratorResult:
        self.resumed.append((assessment_id, decision))
        if self._raise_on_resume is not None:
            raise self._raise_on_resume
        return self._resume_result
