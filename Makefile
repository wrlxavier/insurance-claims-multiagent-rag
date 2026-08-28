# Add Makefile targets: install, lint, format, format-check, typecheck, test, test-integration, check.

.PHONY: install lint format format-check typecheck test test-integration test-eval check migrate migrate-down extract-text remove-boilerplate build-clause-tree parse build-chunks check-embedding-input-length load-chunks embed-chunks benchmark-ann-index sample-parsing-quality validate-parsing-quality-sample score-parsing-quality escalate-vision-boundaries fetch-corpus-artifacts package-corpus-artifacts validate-golden-set draft-golden-questions-casco repair-golden-questions-casco finalize-golden-set-casco draft-golden-questions-adversarial repair-golden-questions-adversarial finalize-golden-set-adversarial draft-synthetic-claims finalize-synthetic-claims validate-synthetic-claims draft-product-claim-mismatch finalize-product-claim-mismatch validate-product-claim-mismatch draft-unanswerable-questions finalize-unanswerable-questions eval-retrieval eval-retrieval-lexical review-golden-set-sample

help:
	@echo "Available targets:"
	@echo "  install           - Install dependencies"
	@echo "  lint              - Run linter"
	@echo "  format            - Format code"
	@echo "  format-check      - Check code formatting"
	@echo "  typecheck         - Run type checker"
	@echo "  test              - Run unit tests"
	@echo "  test-integration  - Apply migrations, then run the database-backed tests in tests/integration (needs TEST_DATABASE_URL)"
	@echo "  test-eval         - Run the eval-marked pytest suite (requires build/parsed_clauses.jsonl; run fetch-corpus-artifacts or parse first)"
	@echo "  check             - Run all checks (lint, format-check, typecheck, test)"
	@echo "  migrate           - Apply Alembic migrations to the configured database"
	@echo "  migrate-down      - Roll back the latest Alembic migration"
	@echo "  extract-text      - Extract and cache text from the policy corpus"
	@echo "  remove-boilerplate - Remove boilerplate from the cached extraction"
	@echo "  build-clause-tree - Recover the clause tree from the cleaned corpus"
	@echo "  parse             - Rebuild the parsed-clause corpus under build/"
	@echo "  build-chunks      - M3-01: chunk the clause tree into build/chunks.{parquet,jsonl} + docs/CHUNKING_REPORT.md"
	@echo "  check-embedding-input-length - M3-02: tokenise build/chunks.jsonl with the pinned embedding model and check every chunk fits its input window"
	@echo "  load-chunks       - M3-02: upsert build/chunks.jsonl into the chunk table (idempotent; needs Postgres + make migrate)"
	@echo "  embed-chunks      - M3-02: embed every chunk still missing a vector with the local sentence-transformers model (depends on load-chunks; runs the optional embed uv group; writes eval/runs/embedding_cost_report.{md,json})"
	@echo "  benchmark-ann-index - M3-02: build the HNSW index against TEST_DATABASE_URL and measure build time, size, exact-vs-ANN latency and filtered result counts (synthetic vectors; writes eval/runs/ann_index_benchmark.{md,json})"
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
	@echo "  draft-synthetic-claims - M2-04: draft candidate synthetic claim narratives into eval/synthetic_claims_draft.csv for review (--dry-run prints selection counts, no LLM calls)"
	@echo "  finalize-synthetic-claims - M2-04: promote approved rows from eval/synthetic_claims_draft.csv into data/synthetic_claims/claims.jsonl"
	@echo "  validate-synthetic-claims - Validate data/synthetic_claims/claims.jsonl against the schema and the parsed corpus"
	@echo "  draft-product-claim-mismatch - M2-05: draft candidate product/claim mismatch narratives into eval/product_claim_mismatch_draft.csv for review (--dry-run prints selection counts, no LLM calls)"
	@echo "  finalize-product-claim-mismatch - M2-05: promote approved rows from eval/product_claim_mismatch_draft.csv into data/synthetic_claims/product_claim_mismatch.jsonl"
	@echo "  validate-product-claim-mismatch - Validate data/synthetic_claims/product_claim_mismatch.jsonl against the schema and the parsed corpus"
	@echo "  draft-unanswerable-questions - M2-05: draft candidate unanswerable golden questions into eval/unanswerable_draft.csv for review (--dry-run prints selection counts, no LLM calls)"
	@echo "  finalize-unanswerable-questions - M2-05: promote approved rows from eval/unanswerable_draft.csv into data/golden_set/unanswerable.jsonl"
	@echo "  eval-retrieval    - M2-06: score Recall@{1,5,10}/MRR/nDCG@10 against the golden set, broken down by question_type/product_line/extraction_mode, plus exclusion-clause recall (built-in random-retriever self-test; writes eval/runs/retrieval_eval_random.{md,json})"
	@echo "  eval-retrieval-lexical - M3-03: score the BM25 lexical retriever (Portuguese tokenisation + Snowball stemming over build/chunks.jsonl, no metadata filter) on the golden set; writes eval/runs/retrieval_eval_lexical.{md,json}"
	@echo "  review-golden-set-sample - Independent second-reviewer pass over a stratified golden-set-v1 sample (--dry-run prints the sample composition, no model calls); writes data/golden_set/review/review_v1.jsonl and eval/runs/golden_set_review_v1.{md,json}"

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
	PYTHONPATH=app/src uv run pytest -m "not integration and not eval"

test-integration:
	bash scripts/run_integration_tests.sh

test-eval:
	PYTHONPATH=app/src uv run pytest -m eval

check: lint format-check typecheck test

migrate:
	PYTHONPATH=app/src uv run alembic upgrade head

migrate-down:
	PYTHONPATH=app/src uv run alembic downgrade -1

extract-text:
	PYTHONPATH=app/src uv run python scripts/extract_text.py

remove-boilerplate:
	PYTHONPATH=app/src uv run python scripts/remove_boilerplate.py

build-clause-tree:
	PYTHONPATH=app/src uv run python scripts/build_clause_tree.py

parse: extract-text remove-boilerplate build-clause-tree
	PYTHONPATH=app/src uv run python scripts/build_corpus.py

build-chunks:
	PYTHONPATH=app/src uv run python scripts/build_chunks.py

check-embedding-input-length:
	PYTHONPATH=app/src uv run python scripts/check_embedding_input_length.py

load-chunks:
	PYTHONPATH=app/src uv run python scripts/load_chunks.py

embed-chunks: load-chunks
	PYTHONPATH=app/src uv run --group embed python scripts/embed_chunks.py

benchmark-ann-index:
	PYTHONPATH=app/src uv run python scripts/benchmark_ann_index.py

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

draft-synthetic-claims:
	PYTHONPATH=app/src uv run python scripts/draft_synthetic_claims.py

finalize-synthetic-claims:
	PYTHONPATH=app/src uv run python scripts/finalize_synthetic_claims_from_review.py

validate-synthetic-claims:
	PYTHONPATH=app/src uv run python scripts/validate_synthetic_claims.py

draft-product-claim-mismatch:
	PYTHONPATH=app/src uv run python scripts/draft_product_claim_mismatch.py

finalize-product-claim-mismatch:
	PYTHONPATH=app/src uv run python scripts/finalize_product_claim_mismatch_from_review.py

validate-product-claim-mismatch:
	PYTHONPATH=app/src uv run python scripts/validate_product_claim_mismatch.py

draft-unanswerable-questions:
	PYTHONPATH=app/src uv run python scripts/draft_unanswerable_questions.py

finalize-unanswerable-questions:
	PYTHONPATH=app/src uv run python scripts/finalize_golden_set_from_review.py \
		--csv eval/unanswerable_draft.csv

eval-retrieval:
	PYTHONPATH=app/src uv run python scripts/eval_retrieval.py

eval-retrieval-lexical:
	PYTHONPATH=app/src uv run python scripts/eval_retrieval.py --retriever lexical

review-golden-set-sample:
	PYTHONPATH=app/src uv run python scripts/review_golden_set_sample.py