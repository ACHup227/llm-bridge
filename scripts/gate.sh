#!/usr/bin/env bash
# scripts/gate.sh — THE green gate. Every change must pass this before it merges, and CI runs the
# SAME script (.github/workflows/ci.yml) — so "green locally" == "green in CI".
#
# Same 4-stage shape as tldr-filter/backend's gate.sh (uv sync / ruff check / ruff format --check /
# mypy / pytest), collapsed to repo root since llm-bridge is one flat package, not a backend+
# frontend split. NEVER weaken a stage to get green; strengthening is welcome.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== gate: llm-bridge =="
uv sync
uv run ruff check .
uv run ruff format --check .
uv run mypy .
uv run pytest

echo "gate: GREEN"
