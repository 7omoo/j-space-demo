#!/usr/bin/env bash
# The checks that need no model, in the order CI runs them: formatting and lint, the page's JS, unit tests.
# Browser tests (E2E) run separately: uv run pytest -m e2e (after `uv run playwright install chromium`).
# scripts/format.sh applies the formatting this checks.
set -euo pipefail
cd "$(dirname "$0")/.."

PRETTIER="prettier@3.9.9"
PY_PATHS=(src tests experiments scripts)
WEB_FILES=("web/**/*.{js,css,html}" "tests/js/*.mjs")
NODE="$(command -v node || true)"

echo "== ruff"
uv run ruff format --check "${PY_PATHS[@]}"
uv run ruff check "${PY_PATHS[@]}"
if [ -n "$NODE" ]; then
  echo "== prettier";  npx --yes "$PRETTIER" --check "${WEB_FILES[@]}"
  echo "== JS syntax"; find web -name "*.js" -not -path "web/data/*" -print0 | xargs -0 -n1 "$NODE" --check
  echo "== JS tests";  "$NODE" --test tests/js/*.test.mjs
else
  echo "== JS: node not found, skipped"
fi
echo "== unit tests"
uv run pytest -q
