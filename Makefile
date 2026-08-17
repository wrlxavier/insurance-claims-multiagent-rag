# Add Makefile targets: install, lint, format, format-check, typecheck, test, test-integration, check.

.PHONY: install lint format format-check typecheck test test-integration check extract-text remove-boilerplate build-clause-tree parse

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
	@echo "  extract-text      - Extract and cache text from the policy corpus"
	@echo "  remove-boilerplate - Remove boilerplate from the cached extraction"
	@echo "  build-clause-tree - Recover the clause tree from the cleaned corpus"
	@echo "  parse             - Rebuild the parsed-clause corpus under build/"

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
	PYTHONPATH=app/src uv run pytest -m "not integration"

test-integration:
	PYTHONPATH=app/src uv run pytest -m integration

check: lint format-check typecheck test

extract-text:
	PYTHONPATH=app/src uv run python scripts/extract_text.py

remove-boilerplate:
	PYTHONPATH=app/src uv run python scripts/remove_boilerplate.py

build-clause-tree:
	PYTHONPATH=app/src uv run python scripts/build_clause_tree.py

parse: extract-text remove-boilerplate build-clause-tree
	PYTHONPATH=app/src uv run python scripts/build_corpus.py