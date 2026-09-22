#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NODE_ONLY=0
COMMAND_ONLY=0
INSTALL_COMMAND=1
RUNTIME_YES=0
COMMAND_DIR="${NATLANG_DEV_BIN_DIR:-$HOME/.local/bin}"

usage() {
  cat <<'EOF'
Usage: scripts/setup_dev.sh [OPTIONS]

Prepare a natlang development checkout and install its `natlang` command.

  --node-only        Skip the Python virtual environment and editable install.
  --command-only     Only install the command; do not install or build dependencies.
  --no-command       Do not install the command.
  --command-dir DIR  Install the command in DIR (default: ~/.local/bin).
  --yes              Approve a verified managed llama.cpp download when needed.
  -h, --help         Show this help.

NATLANG_DEV_BIN_DIR provides the same override as --command-dir. The installer
never replaces an existing command belonging to another checkout or install.
EOF
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --node-only) NODE_ONLY=1; shift ;;
    --command-only) COMMAND_ONLY=1; shift ;;
    --no-command) INSTALL_COMMAND=0; shift ;;
    --yes) RUNTIME_YES=1; shift ;;
    --command-dir)
      [[ "$#" -ge 2 ]] || { echo "setup_dev.sh: --command-dir requires a directory" >&2; exit 2; }
      COMMAND_DIR="$2"
      shift 2
      ;;
    --help|-h) usage; exit 0 ;;
    *) echo "setup_dev.sh: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "$COMMAND_ONLY" -eq 1 && "$INSTALL_COMMAND" -eq 0 ]]; then
  echo "setup_dev.sh: --command-only and --no-command cannot be used together" >&2
  exit 2
fi

COMMAND_PATH="$COMMAND_DIR/natlang"
COMMAND_TARGET="$ROOT/scripts/natlang"
COMMAND_INSTALLED=0

install_command() {
  mkdir -p "$COMMAND_DIR"
  if [[ -e "$COMMAND_PATH" || -L "$COMMAND_PATH" ]]; then
    if [[ -L "$COMMAND_PATH" && "$(readlink "$COMMAND_PATH")" == "$COMMAND_TARGET" ]]; then
      COMMAND_INSTALLED=1
      return
    fi
    cat >&2 <<EOF
setup_dev.sh: $COMMAND_PATH already exists and does not belong to this checkout.
Choose another directory with --command-dir, remove the existing command yourself,
or rerun with --no-command.
EOF
    exit 1
  fi
  ln -s "$COMMAND_TARGET" "$COMMAND_PATH"
  COMMAND_INSTALLED=1
}

if [[ "$COMMAND_ONLY" -eq 0 ]]; then
  command -v node >/dev/null || { echo "setup_dev.sh: Node.js is required" >&2; exit 1; }
  command -v npm >/dev/null || { echo "setup_dev.sh: npm is required" >&2; exit 1; }
  node -e 'const [major,minor]=process.versions.node.split(".").map(Number); if(major<22 || (major===22 && minor<13)){console.error(`natlang needs Node >=22.13; found ${process.versions.node}`); process.exit(1)}'

  echo "==> Installing TypeScript host dependencies"
  npm --prefix "$ROOT/ts-host" ci

  echo "==> Building the TypeScript host and browser bundle"
  npm --prefix "$ROOT/ts-host" run build

  echo "==> Checking the local model runtime"
  RUNTIME_ARGS=(--setup)
  if [[ "$RUNTIME_YES" -eq 1 ]]; then RUNTIME_ARGS+=(--yes); fi
  NATLANG_RUNTIME_HOME="$ROOT/.natlang/runtime" node "$ROOT/ts-host/dist/cli/main.js" "${RUNTIME_ARGS[@]}"

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
fi

if [[ "$INSTALL_COMMAND" -eq 1 ]]; then
  echo "==> Installing the development command"
  install_command
fi

if [[ "$COMMAND_INSTALLED" -eq 1 ]]; then
  cat <<EOF

Development command installed:
  $COMMAND_PATH -> $COMMAND_TARGET
EOF
  case ":$PATH:" in
    *":$COMMAND_DIR:"*) ;;
    *)
      cat <<EOF

Add its directory to your shell PATH, then open a new shell:
  export PATH="$COMMAND_DIR:\$PATH"

Put that export in your shell profile to keep it across sessions.
EOF
      ;;
  esac
fi

if [[ "$COMMAND_ONLY" -eq 0 ]]; then
  cat <<EOF

Development setup complete.
EOF
fi

if [[ "$COMMAND_INSTALLED" -eq 1 ]]; then
  cat <<EOF

Check the installation and run code from any directory:
  natlang --doctor --json
  natlang path/to/program.nl
  natlang path/to/application
EOF
else
  cat <<EOF

Run through the checkout wrapper:
  $COMMAND_TARGET doctor --json
  $COMMAND_TARGET path/to/program.nl
  $COMMAND_TARGET path/to/application
EOF
fi

cat <<EOF
See $ROOT/DEV_SETUP.md for evidence, notebook, log, profile, and test examples.
EOF
