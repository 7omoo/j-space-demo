#!/usr/bin/env bash
# Format the Python (ruff) and the page (Prettier) the way scripts/check.sh expects.
set -euo pipefail
cd "$(dirname "$0")/.."

uv run ruff format src tests experiments scripts
uv run ruff check --fix --select I src tests experiments scripts
npx --yes prettier@3.9.9 --write "web/**/*.{js,css,html}" "tests/js/*.mjs"
