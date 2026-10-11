#!/usr/bin/env bash
# The checks origin/main must pass: the Python suite (repo venv, node on PATH) and the ts-host build and tests.
#
#   scripts/check-main.sh            everything (tens of minutes on the DGX under load)
#   scripts/check-main.sh --quick    ratchets, spec links, ts-host typecheck: a minute; run it before every push
#   scripts/check-main.sh --python | --ts    one side only
#
# Every step runs even after a failure; the summary lists each step and the exit code is 1 if any failed.
# On the DGX the full run loads models and node workers, so run it under the memory ledger:
#   python3 scripts/memory_ledger.py run --unit natlang-check-main-$(date +%H%M%S) --budget-gb 28 --reserve-gb 0 \
#     --oom-policy continue --class experiment --wait 600 --workdir "$PWD" -- \
#     sh -c 'scripts/check-main.sh > /tmp/check-main.log 2>&1; echo exit=$? >> /tmp/check-main.log'
# Environment: NATLANG_PYTHON (default .venv-neuralese/bin/python), CHECK_MAIN_TS_CONCURRENCY (2).
# Node packages come from `npm install` at the workspace root (never `npm ci` while other sessions run).
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
# A linked worktree has no venv of its own: use the main checkout's.
MAIN=$(cd "$ROOT" && cd "$(git rev-parse --git-common-dir)/.." && pwd)
PY=${NATLANG_PYTHON:-$( [ -x "$ROOT/.venv-neuralese/bin/python" ] && echo "$ROOT" || echo "$MAIN")/.venv-neuralese/bin/python}
# Nor its own node packages: npm installs them at the workspace root and in ts-host, so link the main checkout's
# (one missing level leaves e.g. undici or @wllama unresolved and reads as TS2307/TS7006 errors).
# The bundled applications (applications/*) install their own from their lockfiles (scripts/setup_dev.sh).
for dir in . ts-host $(cd "$MAIN" && ls -d applications/*/ 2>/dev/null | sed 's#/$##'); do
  if [ ! -e "$ROOT/$dir/node_modules" ] && [ -d "$MAIN/$dir/node_modules" ] && [ -d "$ROOT/$dir" ]; then
    ln -s "$MAIN/$dir/node_modules" "$ROOT/$dir/node_modules"
  fi
done
export PATH="$HOME/.local/bin:$PATH"
# Python tests must not depend on a warm Hugging Face cache (the ts-host Neuralese tests do use the local tiny model).
HF_EMPTY=$(mktemp -d)
trap 'rm -rf "$HF_EMPTY"' EXIT

quick=0 python=1 ts=1
for arg in "$@"; do
  case $arg in
    --quick) quick=1 ;;
    --python) ts=0 ;;
    --ts) python=0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

results=()
step() {
  local name=$1; shift
  echo "=== $name: $*"
  local start=$SECONDS
  if (cd "$ROOT" && "$@"); then status=ok; else status="FAILED ($?)"; fi
  results+=("$(printf '%-22s %-12s %4ss' "$name" "$status" $((SECONDS - start)))")
}

if [ $python = 1 ]; then
  if [ $quick = 1 ]; then
    step python-ratchets env HF_HOME="$HF_EMPTY" "$PY" -m pytest -q -p no:cacheprovider tests/test_machine_paths_ratchet.py \
      tests/test_duplicate_helpers_ratchet.py tests/test_check_spec_links.py
  else
    # Two processes: one pytest over everything grows past 18 GB.
    step python-tests env HF_HOME="$HF_EMPTY" "$PY" -m pytest -q -p no:cacheprovider tests --ignore=tests/neuralese
    step python-neuralese env HF_HOME="$HF_EMPTY" "$PY" -m pytest -q -p no:cacheprovider tests/neuralese
  fi
fi
if [ $ts = 1 ]; then
  if [ $quick = 1 ]; then
    step ts-check sh -c 'cd ts-host && npm run --silent check'
  else
    step ts-build-node sh -c 'cd ts-host && npm run --silent build:node'
    # A checkout with a browser build keeps it current (rebuilt only when stale: browser-build-freshness.mjs), so the
    # browser tests run against these sources; without one they skip as before.
    step ts-build-browser sh -c 'cd ts-host && if [ -e dist/browser/natlang.js ]; then node scripts/browser-build-freshness.mjs --rebuild; fi'
    step ts-build-apps sh -c 'cd ts-host && npm run --silent build:applications'
    step ts-browser-types sh -c 'cd ts-host && npm run --silent check:browser-types'
    step ts-layering sh -c 'cd ts-host && npm run --silent check:layering'
    step ts-tests sh -c "cd ts-host && node scripts/test-files.mjs --concurrency ${CHECK_MAIN_TS_CONCURRENCY:-2}"
    step ts-conformance sh -c 'cd ts-host && npm run --silent test:conformance'
  fi
fi

echo "=== summary ($(git -C "$ROOT" rev-parse --short HEAD))"
printf '%s\n' "${results[@]}"
for line in "${results[@]}"; do case $line in *FAILED*) exit 1 ;; esac; done
exit 0
