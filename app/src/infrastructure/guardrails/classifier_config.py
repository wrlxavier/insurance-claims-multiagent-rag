"""The pinned prompt-injection-classifier contract -- [M5-08 Appendix].

Single source of truth for the classifier model, its revision and the
label its own taxonomy uses for a positive prediction -- the same rationale
``infrastructure.rag.embedding_config`` states for the embedder: the model
id, revision and label together determine the exact scores behind a
published false-positive/detection-rate number, so they are module
constants, not `.env` knobs. ``PROMPT_INJECTION_CLASSIFIER_MODEL`` stays in
`.env` as the human-facing name and is cross-checked against
``CLASSIFIER_MODEL_ID`` by a test, exactly as ``EMBEDDING_MODEL`` is against
``EMBEDDING_MODEL_ID``.

Rationale and the real benchmark: docs/PROMPT_INJECTION_CLASSIFIER.md.
"""

# protectai/deberta-v3-base-prompt-injection-v2 -- a
# `DebertaV2ForSequenceClassification` binary classifier (base model:
# microsoft/deberta-v3-base), the current
# major version of protectai's widely-used open prompt-injection detector.
# Standard `transformers` architecture: no `trust_remote_code`, unlike the
# pinned embedder/reranker. Trained entirely on English-language datasets
# (its model card lists `language: en` and no Portuguese corpus) -- worth
# stating plainly, since this project's clause/claim text is Portuguese; see
# the false-positive-rate result in docs/PROMPT_INJECTION_CLASSIFIER.md.
CLASSIFIER_MODEL_ID = "protectai/deberta-v3-base-prompt-injection-v2"

# Pinned by Hub commit -- NOT the floating ``main`` alias -- so a provider-side
# re-upload cannot silently change the weights behind an already-published
# number. Commit last modified 2026-07-09; re-confirm before bumping.
CLASSIFIER_MODEL_REVISION = "90c9989b1a342275dd0d1a95aad283c04e075671"

# config.json: 512-token context window (far shorter than the embedder's/
# reranker's 8192 -- a DeBERTa-v3-base classifier head, not a retrieval
# model). Passed as the tokenizer's ``max_length`` so a long clause excerpt
# or claim narrative is truncated deterministically rather than by a silent
# default.
CLASSIFIER_MAX_INPUT_TOKENS = 512

# config.json's own id2label: {0: "SAFE", 1: "INJECTION"}. The label whose
# score `LocalPromptInjectionClassifier.classify` reports.
CLASSIFIER_POSITIVE_LABEL = "INJECTION"
