#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
revision=b741d505ae9d1e3f28bd4ff4d0227b58b4921ed4
source_path="${1:-vendor/PAROL6-python-API}"
if [[ ! -d "$source_path" ]]; then
  git clone https://github.com/PCrnjak/PAROL6-python-API.git "$source_path"
  git -C "$source_path" checkout "$revision"
fi
[[ "$(git -C "$source_path" rev-parse HEAD)" == "$revision" ]] || { echo 'Upstream revision mismatch'; exit 1; }
[[ -z "$(git -C "$source_path" status --porcelain)" ]] || { echo 'Use a clean pinned checkout'; exit 1; }
uv venv --python 3.12 --allow-existing .upstream-venv
uv pip install --python .upstream-venv/bin/python --constraint requirements/upstream-constraints.txt -e "$source_path" -e .
.upstream-venv/bin/humaned-lab upstream-smoke
