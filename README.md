# insurance-claims-multiagent-rag
Multi-agent assistant that checks insurance claims against real Brazilian policy conditions (SUSEP). LangGraph conditional graph with parallel agents, hybrid RAG with reranking, human-in-the-loop checkpoint and full audit trail. FastAPI + Clean Architecture. Retrieval and end-to-end quality measured on a hand-curated golden set.

The MIT license covers the source code in this repository only.
Documents under `data/policies/raw/` are published by their respective
insurers and remain their property — see NOTICE.md.

## Development Commands

### Pre-commit hooks

1. Install the dev dependency (skip if already in the lockfile — run `uv sync` instead):

```bash
uv add --dev pre-commit
```

2. Enable the hooks in your local clone:

```bash
uv run pre-commit install
```

3. Run the hooks against all files once, to check the existing codebase:

```bash
uv run pre-commit run --all-files
```
