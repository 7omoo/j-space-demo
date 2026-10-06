#!/usr/bin/env bash
# Clone the upstream jacobian-lens reference implementation at the pinned commit, then sync the uv env.
# Idempotent: re-running only verifies the checkout and re-syncs.
set -euo pipefail

UPSTREAM_URL="https://github.com/anthropics/jacobian-lens"
UPSTREAM_COMMIT="581d398613e5602a5af361e1c34d3a92ea82ba8e"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENDOR="$ROOT/vendor/jacobian-lens"

if [ ! -d "$VENDOR/.git" ]; then
  git clone --quiet "$UPSTREAM_URL" "$VENDOR"
fi
git -C "$VENDOR" checkout --quiet "$UPSTREAM_COMMIT"

head="$(git -C "$VENDOR" rev-parse HEAD)"
if [ "$head" != "$UPSTREAM_COMMIT" ]; then
  echo "vendor/jacobian-lens is at $head, expected $UPSTREAM_COMMIT" >&2
  exit 1
fi
if [ -n "$(git -C "$VENDOR" status --porcelain)" ]; then
  echo "vendor/jacobian-lens has local modifications; upstream must stay unedited" >&2
  exit 1
fi
echo "upstream pinned at $head"

latest="$(git ls-remote "$UPSTREAM_URL" HEAD | cut -f1)"
[ "$latest" = "$UPSTREAM_COMMIT" ] || echo "note: upstream HEAD is now $latest (staying on the pinned commit)"

cd "$ROOT" && uv sync
