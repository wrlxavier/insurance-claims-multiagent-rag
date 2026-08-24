# Add Makefile targets: install, lint, format, format-check, typecheck, test, test-integration, check.

.PHONY: install lint format format-check typecheck test test-integration check extract-text remove-boilerplate build-clause-tree parse sample-parsing-quality validate-parsing-quality-sample score-parsing-quality escalate-vision-boundaries fetch-corpus-artifacts package-corpus-artifacts validate-golden-set draft-golden-questions-casco repair-golden-questions-casco finalize-golden-set-casco draft-golden-questions-adversarial repair-golden-questions-adversarial finalize-golden-set-adversarial

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
	@echo "  sample-parsing-quality - Draw the M1-08 stratified 50-clause sample"
	@echo "  validate-parsing-quality-sample - Automated LLM validation of the M1-08b sample (fills judgment columns)"
	@echo "  score-parsing-quality  - Score the annotated M1-08 sample and write eval/parsing_quality_results.md"
	@echo "  escalate-vision-boundaries - M1-04d: vision-LLM boundary review of suspicious clauses (opt-in, not part of parse)"
	@echo "  fetch-corpus-artifacts - Download the pre-computed corpus/LLM caches instead of running make parse"
	@echo "  package-corpus-artifacts - Maintainer-only: build the release tarball fetch-corpus-artifacts downloads"
	@echo "  validate-golden-set - Validate data/golden_set/*.jsonl against the schema and the parsed corpus"
	@echo "  draft-golden-questions-casco - M2-02: draft candidate golden questions over the 15 CASCO documents into eval/golden_set_draft_casco.csv for review (overwrites that file; use repair- once rows are finalized)"
	@echo "  repair-golden-questions-casco - M2-02: re-draft/complete the CASCO draft using the author's review verdicts (requires REVIEW=<csv>)"
	@echo "  finalize-golden-set-casco - M2-02: promote approved rows from eval/golden_set_draft_casco.csv into data/golden_set/*.jsonl"
	@echo "  draft-golden-questions-adversarial - M2-03: draft candidate adversarial golden questions (coverage_with_exclusion, cross_document, hdi_brand_collision, bundle_section) into eval/golden_set_draft_adversarial.csv for review"
	@echo "  repair-golden-questions-adversarial - M2-03: re-draft the adversarial draft using the author's review verdicts (requires REVIEW=<csv>)"
	@echo "  finalize-golden-set-adversarial - M2-03: promote approved rows from eval/golden_set_draft_adversarial.csv into data/golden_set/*.jsonl"

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

fetch-corpus-artifacts:
	PYTHONPATH=app/src uv run python scripts/fetch_corpus_artifacts.py

package-corpus-artifacts:
	PYTHONPATH=app/src uv run python scripts/package_corpus_artifacts.py

validate-golden-set:
	PYTHONPATH=app/src uv run python scripts/validate_golden_set.py

draft-golden-questions-casco:
	PYTHONPATH=app/src uv run python scripts/draft_golden_questions_casco.py

# The review CSV is an input the author writes per round, so there is no
# sensible default: name it explicitly.
REVIEW ?=

repair-golden-questions-casco:
	@test -n "$(REVIEW)" || { \
		echo "REVIEW is required: make $@ REVIEW=eval/<your-review>.csv"; \
		exit 1; \
	}
	PYTHONPATH=app/src uv run python scripts/draft_golden_questions_casco.py \
		--review-csv $(REVIEW)

finalize-golden-set-casco:
	PYTHONPATH=app/src uv run python scripts/finalize_golden_set_from_review.py

draft-golden-questions-adversarial:
	PYTHONPATH=app/src uv run python scripts/draft_golden_questions_adversarial.py

repair-golden-questions-adversarial:
	@test -n "$(REVIEW)" || { \
		echo "REVIEW is required: make $@ REVIEW=eval/<your-review>.csv"; \
		exit 1; \
	}
	PYTHONPATH=app/src uv run python scripts/draft_golden_questions_adversarial.py \
		--review-csv $(REVIEW)

finalize-golden-set-adversarial:
	PYTHONPATH=app/src uv run python scripts/finalize_golden_set_from_review.py \
		--csv eval/golden_set_draft_adversarial.csv