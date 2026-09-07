"""The one edge where domain/application errors become HTTP responses -- [M5-04].

Every failure a use case raises is caught here and rendered as the uniform
envelope ``{"error": {"code", "message", "details"}}`` with a stable string
code, so a client can branch on ``code`` and never on a status alone or a
message string. The ``application.errors`` / ``domain.errors`` hierarchies were
built as a single cluster for exactly this (``application/errors.py`` docstring).

Registration is by exception type; Starlette resolves a raised exception to the
most-derived registered handler via its MRO, so the order of the table does not
matter. ``RequestValidationError`` (FastAPI's own body-validation failure) is
normalised into the same envelope; a genuinely unexpected exception becomes a
logged 500 with no traceback in the body.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from application.errors import (
    ApplicationError,
    AssessmentAlreadyDecidedError,
    AssessmentNotFoundError,
    OrchestratorContractError,
    UnknownClauseError,
)
from domain.errors import (
    CitationRequiredError,
    DecisionMustReferenceAssessmentError,
    DomainError,
    InvalidCnpjError,
    InvalidSusepProcessError,
    InvalidValueObjectError,
    InvariantViolationError,
    PolicyYearMismatchError,
    VerdictNotPermittedError,
)
from presentation.schemas import ErrorBody, ErrorResponse

logger = logging.getLogger(__name__)

_Handler = Callable[[Request, Exception], Awaitable[JSONResponse]]

# (exception type, HTTP status, stable error code). Leaf types plus the three
# bases plus `ValueError` -- Starlette picks the most-derived match.
_MAPPING: tuple[tuple[type[Exception], int, str], ...] = (
    (AssessmentNotFoundError, 404, "assessment_not_found"),
    (AssessmentAlreadyDecidedError, 409, "assessment_already_decided"),
    (UnknownClauseError, 422, "unknown_clause"),
    (OrchestratorContractError, 502, "orchestrator_contract_error"),
    (ApplicationError, 500, "application_error"),
    (CitationRequiredError, 422, "citation_required"),
    (VerdictNotPermittedError, 422, "verdict_not_permitted"),
    (DecisionMustReferenceAssessmentError, 422, "decision_must_reference_assessment"),
    (PolicyYearMismatchError, 422, "policy_year_mismatch"),
    (InvalidSusepProcessError, 422, "invalid_susep_process"),
    (InvalidCnpjError, 422, "invalid_cnpj"),
    (InvalidValueObjectError, 422, "invalid_value_object"),
    (InvariantViolationError, 422, "invariant_violation"),
    (DomainError, 422, "domain_error"),
    (ValueError, 422, "invalid_request"),
)


def _envelope(
    status_code: int,
    code: str,
    message: str,
    details: dict[str, object] | None = None,
) -> JSONResponse:
    body = ErrorResponse(error=ErrorBody(code=code, message=message, details=details))
    return JSONResponse(status_code=status_code, content=body.model_dump(mode="json"))


def _details_for(exc: Exception) -> dict[str, object] | None:
    if isinstance(
        exc,
        AssessmentNotFoundError | AssessmentAlreadyDecidedError | CitationRequiredError,
    ):
        return {"assessment_id": exc.assessment_id}
    if isinstance(exc, UnknownClauseError):
        return {"clause_ids": list(exc.clause_ids)}
    return None


def _make_handler(status_code: int, code: str) -> _Handler:
    async def handler(_request: Request, exc: Exception) -> JSONResponse:
        return _envelope(status_code, code, str(exc), _details_for(exc))

    return handler


async def _validation_handler(_request: Request, exc: Exception) -> JSONResponse:
    errors = exc.errors() if isinstance(exc, RequestValidationError) else []
    return _envelope(
        422,
        "request_validation_error",
        "request body failed validation",
        {"errors": list(errors)},
    )


async def _unexpected_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception(
        "unhandled error on %s %s", request.method, request.url.path, exc_info=exc
    )
    return _envelope(500, "internal_error", "an unexpected error occurred")


def register_exception_handlers(app: FastAPI) -> None:
    """Wire every domain/application error to its HTTP status and stable code."""
    for exc_type, status_code, code in _MAPPING:
        app.add_exception_handler(exc_type, _make_handler(status_code, code))
    app.add_exception_handler(RequestValidationError, _validation_handler)
    app.add_exception_handler(Exception, _unexpected_handler)
