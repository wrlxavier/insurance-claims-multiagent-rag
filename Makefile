# Add Makefile targets: install, lint, format, format-check, typecheck, test, test-integration, check.

.PHONY: install lint format format-check typecheck test test-integration test-eval check migrate migrate-down extract-text remove-boilerplate build-clause-tree parse build-chunks check-embedding-input-length load-chunks embed-chunks build-index benchmark-ann-index benchmark-ann-index-real sample-parsing-quality validate-parsing-quality-sample score-parsing-quality escalate-vision-boundaries fetch-corpus-artifacts package-corpus-artifacts validate-golden-set draft-golden-questions-casco repair-golden-questions-casco finalize-golden-set-casco draft-golden-questions-adversarial repair-golden-questions-adversarial finalize-golden-set-adversarial draft-synthetic-claims finalize-synthetic-claims validate-synthetic-claims draft-product-claim-mismatch finalize-product-claim-mismatch validate-product-claim-mismatch draft-unanswerable-questions finalize-unanswerable-questions eval-retrieval eval-retrieval-lexical eval-retrieval-dense eval-retrieval-hybrid eval-retrieval-rerank eval-retrieval-co-retrieval eval-retrieval-matrix eval-insufficient-context-gate eval-intake tune-reranking tune-exclusion-co-retrieval review-golden-set-sample

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
	@echo "  build-index       - M3-08: end-to-end reproducible index build: raw PDFs -> parse -> chunk -> Postgres -> embeddings (composes parse/build-chunks/migrate/load-chunks/embed-chunks; needs the same LLM_* env + tesseract as make parse, and a running Postgres; the parse stage is skipped when build/parsed_clauses.jsonl already exists)"
	@echo "  benchmark-ann-index - M3-02: build the HNSW index against TEST_DATABASE_URL and measure build time, size, exact-vs-ANN latency and filtered result counts (synthetic vectors; writes eval/runs/ann_index_benchmark.{md,json})"
	@echo "  benchmark-ann-index-real - M3-08: the ANN-earns-its-place A/B on the REAL embeddings in DATABASE_URL - HNSW recall@10 vs exact + latency, all inside a rolled-back transaction (runs the optional embed uv group; writes eval/runs/ann_index_benchmark_real.{md,json})"
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
	@echo "  eval-retrieval-dense - M3-04: score the dense pgvector retriever with the SUSEP-process+CNPJ pre-filter (needs a running Postgres with loaded+embedded chunks; runs the optional embed uv group); writes eval/runs/retrieval_eval_dense_filter-default.{md,json}"
	@echo "  eval-retrieval-hybrid - M3-04: score the hybrid (RRF fusion of lexical+dense) retriever with the SUSEP-process+CNPJ pre-filter (same Postgres + embed-group requirement as eval-retrieval-dense; pass --fusion weighted / --filter none on the script for the comparison); writes eval/runs/retrieval_eval_hybrid_*.{md,json}. Comparison and verdict: docs/HYBRID_RETRIEVAL.md"
	@echo "  eval-retrieval-rerank - M3-05: score hybrid RRF + cross-encoder rerank with the SUSEP-process+CNPJ pre-filter at the chosen candidate depth (same Postgres + embed-group requirement); writes eval/runs/retrieval_eval_hybrid_rrf_rerank_filter-default.{md,json}. Verdict: docs/RERANKING.md"
	@echo "  eval-retrieval-co-retrieval - M3-06: score hybrid RRF + rerank + exclusion co-retrieval with the SUSEP-process+CNPJ pre-filter (same Postgres + embed-group requirement); reserves top-k slots for the exclusion clauses linked to each retrieved coverage clause; writes eval/runs/retrieval_eval_hybrid_rrf_rerank_co-retrieval_filter-default.{md,json}. Rule: docs/ARCHITECTURE.md; measurement: docs/EXCLUSION_CO_RETRIEVAL.md"
	@echo "  eval-retrieval-matrix - M3-08: run every retrieval configuration (lexical / dense / hybrid RRF / +rerank / +co-retrieval / weighted+rerank+co-retrieval) on the SUSEP-process+CNPJ path in one pass, with per-query latency (same Postgres + embed-group requirement); writes eval/runs/retrieval_benchmark_matrix.{md,json}. Committed table and verdict: docs/RETRIEVAL_BENCHMARK.md"
	@echo "  eval-insufficient-context-gate - M3-07: calibrate the insufficient-context gate on retrieval signals over the 23 unanswerable golden questions (hybrid RRF + rerank, SUSEP-process+CNPJ filter; same Postgres + embed-group requirement); sweeps the abstain threshold, reports precision/recall + the false-negative cases; writes eval/runs/insufficient_context_gate.{md,json} + the committed snapshot eval/insufficient_context_gate_signals.json. Verdict: docs/INSUFFICIENT_CONTEXT_GATE.md"
	@echo "  eval-intake       - M4-02: run the intake node over every synthetic claim with the real fast model and report product-line accuracy, missing-info recall/false-positives and field population (needs LLM_* in .env); writes eval/runs/intake_extraction.{md,json} + a per-claim predictions JSONL. Analysis: docs/INTAKE_EXTRACTION.md"
	@echo "  tune-reranking    - M3-05: sweep the reranker candidate depth on the golden set and record the metrics + latency curve (same Postgres + embed-group requirement); writes eval/runs/rerank_tuning.{md,json}. Curve and chosen depth: docs/RERANKING.md"
	@echo "  tune-exclusion-co-retrieval - M3-06: sweep the reserved exclusion-slot count on the golden set (pure-Python replay over one cached rerank pass; same Postgres + embed-group requirement for that pass); writes eval/runs/exclusion_co_retrieval_tuning.{md,json}. Curve and chosen count: docs/EXCLUSION_CO_RETRIEVAL.md"
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

# M3-08: the whole index, reproducibly, from raw PDFs. Needs the same
# environment as `make parse` (LLM_* in .env, tesseract) plus a running
# Postgres. The file prerequisite skips the expensive parse stage (OCR + LLM
# classification + vision escalation) when build/parsed_clauses.jsonl is already
# there; build-chunks still runs but is cache-served, and load-chunks /
# embed-chunks are idempotent / short-circuiting, so a re-run is cheap.
build/parsed_clauses.jsonl:
	$(MAKE) parse

build-index: build/parsed_clauses.jsonl build-chunks check-embedding-input-length migrate load-chunks embed-chunks
	@echo "build-index: raw -> parsed clauses -> chunks -> Postgres -> embeddings. Index ready for 'make eval-retrieval-matrix'."

benchmark-ann-index:
	PYTHONPATH=app/src uv run python scripts/benchmark_ann_index.py

benchmark-ann-index-real:
	PYTHONPATH=app/src uv run --group embed python -m scripts.benchmark_ann_index --real-embeddings

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

eval-retrieval-dense:
	PYTHONPATH=app/src uv run --group embed python scripts/eval_retrieval.py --retriever dense --filter default

eval-retrieval-hybrid:
	PYTHONPATH=app/src uv run --group embed python scripts/eval_retrieval.py --retriever hybrid --filter default

eval-retrieval-rerank:
	PYTHONPATH=app/src uv run --group embed python scripts/eval_retrieval.py --retriever hybrid --filter default --rerank

eval-retrieval-co-retrieval:
	PYTHONPATH=app/src uv run --group embed python scripts/eval_retrieval.py --retriever hybrid --filter default --rerank --co-retrieval

eval-retrieval-matrix:
	PYTHONPATH=app/src uv run --group embed python -m scripts.benchmark_retrieval_matrix

eval-insufficient-context-gate:
	PYTHONPATH=app/src uv run --group embed python -m scripts.eval_insufficient_context_gate

eval-intake:
	PYTHONPATH=app/src uv run python -m scripts.eval_intake

tune-reranking:
	PYTHONPATH=app/src uv run --group embed python -m scripts.tune_reranking

tune-exclusion-co-retrieval:
	PYTHONPATH=app/src uv run --group embed python -m scripts.tune_exclusion_co_retrieval

review-golden-set-sample:
	PYTHONPATH=app/src uv run python scripts/review_golden_set_sample.py