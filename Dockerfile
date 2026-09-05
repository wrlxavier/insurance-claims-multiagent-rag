# [M5-09]. One image for both the `api` and `worker` Compose services -- they
# share the same composition root (infrastructure.bootstrap.build_core_components)
# and therefore the same dependency set; `worker` just overrides `command:` to
# run scripts.run_assessment_worker instead of uvicorn. See docs/DEPLOYMENT.md.
#
# Tesseract / the OCR path are deliberately NOT installed here: they belong to
# the offline parsing pipeline (`make parse`), which never runs inside these
# containers -- the graph reads an already-populated Postgres, not raw PDFs.

FROM python:3.12-slim AS builder

# build-essential: some of the `embed` group's transitive C-extension deps
# (torch/transformers) may need to build from source depending on the
# resolved wheel for the target platform.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Pinned to the same uv release this project is developed with (`uv --version`).
COPY --from=ghcr.io/astral-sh/uv:0.11.29 /uv /uvx /usr/local/bin/

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# Dependencies first, so an app-code-only change doesn't invalidate this layer.
# `embed` is the optional group that carries sentence-transformers/transformers
# (torch) -- required at runtime because the retrieval stack loads the
# embedder and cross-encoder in-process (docs/EMBEDDINGS.md, docs/RERANKING.md).
# `--no-default-groups` drops `dev` (ruff/mypy/pytest/...), which the running
# service never needs.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project --no-default-groups --group embed

COPY app/src ./app/src
COPY scripts ./scripts
COPY alembic ./alembic
COPY alembic.ini ./
# The one runtime dependency under data/: a small, committed CSV the lexical
# retriever's stemmer reads at startup (docs/LEXICAL_RETRIEVAL.md). Everything
# else under data/ is parsing-only or a self-healing cache -- see .dockerignore.
COPY data/rag ./data/rag
RUN uv sync --locked --no-default-groups --group embed

FROM python:3.12-slim AS runtime

# curl: the `api` healthcheck. procps (pgrep): the `worker` healthcheck --
# it has no HTTP surface, so liveness is "the process is running".
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl procps \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /app /app

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONPATH="/app/app/src" \
    PYTHONUNBUFFERED=1

EXPOSE 8000

# `worker` overrides this in compose.yaml.
CMD ["uvicorn", "presentation.app:app", "--host", "0.0.0.0", "--port", "8000"]
