# Add Makefile targets: install, lint, format, format-check, typecheck, test, test-integration, check.

.PHONY: install lint format format-check typecheck test test-integration check extract-text remove-boilerplate build-clause-tree parse sample-parsing-quality validate-parsing-quality-sample score-parsing-quality escalate-vision-boundaries

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
	@echo "  sample-parsing-quality - Draw the M1-08 stratified 50-clause sample for manual review"
	@echo "  validate-parsing-quality-sample - Automated LLM validation of the M1-08b sample (fills judgment columns)"
	@echo "  score-parsing-quality  - Score the annotated M1-08 sample and write eval/parsing_quality_results.md"
	@echo "  escalate-vision-boundaries - M1-04d: vision-LLM boundary review of suspicious clauses (opt-in, not part of parse)"

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

sample-parsing-quality:
	PYTHONPATH=app/src uv run python scripts/sample_parsing_quality.py

validate-parsing-quality-sample:
	PYTHONPATH=app/src uv run python scripts/validate_parsing_quality_sample.py

score-parsing-quality:
	PYTHONPATH=app/src uv run python scripts/score_parsing_quality.py

escalate-vision-boundaries:
	PYTHONPATH=app/src uv run python scripts/escalate_vision_boundaries.py