# Add Makefile targets: install, lint, format, format-check, typecheck, test, test-integration, check.

.PHONY: install lint format format-check typecheck test test-integration check

help:
	@echo "Available targets:"
	@echo "  install           - Install dependencies"
	@echo "  lint              - Run linter"
	@echo "  format            - Format code"
	@echo "  format-check      - Check code formatting"
	@echo "  typecheck         - Run type checker"
	@echo "  test              - Run unit tests"
	@echo "  test-integration  - Run integration tests"
	@echo "  check             - Run all checks (lint, format-check, typecheck, test)"

install:
	uv sync

lint:
	uv run ruff check .

format:
	uv run ruff format .

format-check:
	uv run ruff format --check .

typecheck:
	uv run mypy --strict app/src

test:
	PYTHONPATH=app/src uv run pytest

test-integration:
	@echo "Integration tests not implemented yet."

check: lint format-check typecheck test