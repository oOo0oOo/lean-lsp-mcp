#!/usr/bin/env sh
set -eu

PROJECT_ROOT="${LEAN_PROJECT_PATH:-/workspace}"
if [ ! -d "$PROJECT_ROOT" ]; then
  echo "warning: LEAN_PROJECT_PATH '$PROJECT_ROOT' does not exist in the container" >&2
fi

exec lean-lsp-mcp "$@"
