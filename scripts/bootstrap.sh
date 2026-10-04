#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
command -v uv >/dev/null || { echo 'Install uv from https://docs.astral.sh/uv/getting-started/installation/'; exit 1; }
uv python install 3.12
extras=(--extra train --extra hub)
if [[ "${1:-}" == "--core-only" ]]; then extras=(); fi
if [[ "${1:-}" == "--camera" ]]; then extras+=(--extra camera); fi
uv sync --frozen --python 3.12 --group dev "${extras[@]}"
.venv/bin/humaned-lab doctor
echo 'Ready. Start: .venv/bin/humaned-lab serve'
