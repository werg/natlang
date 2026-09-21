#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NODE_ONLY=0

usage() {
  cat <<'EOF'
Usage: scripts/setup_dev.sh [--node-only]

Build the TypeScript host and optionally create the Python development
environment.

  --node-only  Skip the Python virtual environment and editable install.
EOF
}

for arg in "$@"; do
  case "$arg" in
    --node-only) NODE_ONLY=1 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "setup_dev.sh: unknown argument: $arg" >&2; usage >&2; exit 2 ;;
  esac
done

command -v node >/dev/null || { echo "setup_dev.sh: Node.js is required" >&2; exit 1; }
command -v npm >/dev/null || { echo "setup_dev.sh: npm is required" >&2; exit 1; }
node -e 'const [major,minor]=process.versions.node.split(".").map(Number); if(major<22 || (major===22 && minor<13)){console.error(`natlang needs Node >=22.13; found ${process.versions.node}`); process.exit(1)}'

echo "==> Installing TypeScript host dependencies"
npm --prefix "$ROOT/ts-host" ci

echo "==> Building the TypeScript host and browser bundle"
npm --prefix "$ROOT/ts-host" run build

if [[ "$NODE_ONLY" -eq 0 ]]; then
  command -v uv >/dev/null || {
    echo "setup_dev.sh: uv is required for the Python development environment; install uv or use --node-only" >&2
    exit 1
  }
  echo "==> Preparing the Python development environment"
  if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
    uv venv --python 3.12 "$ROOT/.venv"
  fi
  uv pip install --python "$ROOT/.venv/bin/python" -e "$ROOT[js,dev]"
fi

cat <<EOF

Development setup complete.

Check the installation:
  $ROOT/scripts/natlang doctor --json

Run a program or application directly from its path:
  $ROOT/scripts/natlang run path/to/program.nl
  $ROOT/scripts/natlang-app path/to/app
  $ROOT/scripts/natlang app run packages/semantic-terminal.natlang.json

See $ROOT/DEV_SETUP.md for evidence, notebook, log, profile, and test examples.
EOF
