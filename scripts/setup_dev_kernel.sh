#!/usr/bin/env bash
#
# scripts/setup_dev_kernel.sh
#
# Dev-only tooling: adds ipykernel as a dev dependency (via uv) and registers
# a Jupyter kernel bound to this project's virtual environment, so notebooks
# in notebooks/scratch/ run against the same environment as the rest of the
# codebase instead of whatever kernel happens to be installed globally.
#
# Usage:
#   ./scripts/setup_dev_kernel.sh

set -euo pipefail

PROJECT_NAME="insurance-claims-multiagent-rag"
KERNEL_NAME="${PROJECT_NAME}"
DISPLAY_NAME="Insurance Claims (uv)"

if [ ! -f "pyproject.toml" ]; then
  echo "Error: run this script from the repository root (pyproject.toml not found)." >&2
  exit 1
fi

if ! command -v uv &> /dev/null; then
  echo "Error: uv is not installed. See https://docs.astral.sh/uv/ for installation." >&2
  exit 1
fi

echo "==> Adding ipykernel as a dev dependency..."
uv add --dev ipykernel

echo "==> Registering Jupyter kernel '${KERNEL_NAME}'..."
uv run python -m ipykernel install --user \
  --name "${KERNEL_NAME}" \
  --display-name "${DISPLAY_NAME}"

echo "==> Done."
echo "    Select '${DISPLAY_NAME}' as the kernel in Jupyter/VS Code, or run:"
echo "    uv run jupyter lab"
echo "    from the repo root to launch directly inside the project's environment."