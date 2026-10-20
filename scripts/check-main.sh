#!/usr/bin/env bash
# The checks origin/main must pass: the Python suite (repo venv, node on PATH) and the ts-host build and tests.
#
#   scripts/check-main.sh            everything (tens of minutes on the DGX under load)
#   scripts/check-main.sh --quick    ratchets, spec links, ts-host typecheck: a minute; run it before every push
#   scripts/check-main.sh --python | --ts    one side only
#
# Every step runs even after a failure; the summary lists each step and the exit code is 1 if any failed.
# On the DGX the full run loads models and node workers, so run it under the memory ledger, with the Neuralese TS tests'
# Python server on the GPU (on the CPU beside training it holds 10-16 GB and does not finish; on the GPU the whole run
# peaks at ~12.2 GB, in python-neuralese-gpu, 2026-10-12):
#   python3 scripts/memory_ledger.py run --unit natlang-check-main-$(date +%H%M%S) --budget-gb 13 --reserve-gb 2 \
#     --oom-policy continue --class experiment --wait 600 --workdir "$PWD" -- \
#     sh -c 'NATLANG_NEURALESE_DEVICE=cuda scripts/check-main.sh > /tmp/check-main.log 2>&1; echo exit=$? >> /tmp/check-main.log'
# Environment: NATLANG_PYTHON overrides discovery; otherwise prefer .venv-neuralese, then .venv
# in this checkout or its main worktree. CHECK_MAIN_TS_CONCURRENCY defaults to 2.
# On Pop, the pinned training image supplies Python 3.12/PyTorch/Transformers:
#   run this under memory_ledger.py --container <name>, setting
#   NATLANG_PYTHON=$PWD/scripts/python_training_image.sh,
#   NATLANG_PYTHON_CONTAINER=<name>, NATLANG_PYTHON_IMAGE=<installed image>.
# Node packages come from `npm install` at the workspace root (never `npm ci` while other sessions run).
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
# A linked worktree has no venv of its own: use the main checkout's.
MAIN=$(cd "$ROOT" && cd "$(git rev-parse --git-common-dir)/.." && pwd)
# Share installed third-party packages with linked worktrees while keeping
# workspace package links pointed at the checkout under test. A single root
# node_modules symlink resolves @natlang/* to the wrong source revision.
python3 "$ROOT/scripts/prepare_check_main_node_modules.py" "$ROOT" "$MAIN"
# The ts-host and application installs contain their third-party dependencies;
# link those trees when the worktree does not already have its own install.
# (One missing level leaves e.g. undici or @wllama unresolved and reads as TS2307/TS7006 errors.)
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
# Disposable fixture identities must not enlarge the production hash index.
# See CHECK_MAIN_HASH_CACHE_DECOMPOSITION_20261020.md.
export NATLANG_HASH_CACHE="${NATLANG_HASH_CACHE:-$HF_EMPTY/natlang-sha256-cache.jsonl}"

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

step source-review-manifest python3 scripts/source_review.py verify

python_setup_error() { printf '%s\n' "$1" >&2; return 2; }

# The Python suite has integration tests that invoke ts-host/dist/teacher/collector.js.
# In a fresh worktree that output does not exist until the Node build runs, so
# make the artifact prerequisite explicit before either full suite starts.
# This is also required for --python; it does not turn a missing build into a
# skipped test or a passing result.
node_build_done=0
if [ $quick = 0 ] && { [ $python = 1 ] || [ $ts = 1 ]; }; then
  step ts-build-node sh -c 'cd ts-host && npm run --silent build:node'
  node_build_done=1
fi

if [ $python = 1 ]; then
  python_error=
  if [ -n "${NATLANG_PYTHON:-}" ]; then
    PY=$NATLANG_PYTHON
    if ! command -v -- "$PY" >/dev/null 2>&1; then
      python_error="Python checks requested, but NATLANG_PYTHON does not resolve to an executable: $PY"
    fi
  else
    PY=
    for candidate in \
      "$ROOT/.venv-neuralese/bin/python" \
      "$MAIN/.venv-neuralese/bin/python" \
      "$ROOT/.venv/bin/python" \
      "$MAIN/.venv/bin/python" \
      /usr/bin/python3; do
      # A stale general-purpose venv may exist without the neuralese test
      # dependencies. Select an interpreter that can actually run this suite.
      if [ "$quick" = 1 ]; then
        interpreter_probe='import pytest, torch'
      else
        interpreter_probe='import sys; assert sys.version_info >= (3, 11); import pytest, torch, transformers; from transformers import Lfm2Config; from transformers.modeling_rope_utils import dynamic_rope_update'
      fi
      if [ -x "$candidate" ] && "$candidate" -c "$interpreter_probe" >/dev/null 2>&1; then
        PY=$candidate
        break
      fi
    done
    if [ -z "$PY" ]; then
      python_error="Python checks requested, but no compatible Python >=3.11 with pytest, torch and the required Transformers API was found. Set NATLANG_PYTHON or install them in .venv-neuralese."
    fi
  fi
  if [ -n "$python_error" ]; then
    step python-setup python_setup_error "$python_error"
  elif [ $quick = 1 ]; then
    step python-ratchets env HF_HOME="$HF_EMPTY" "$PY" -m pytest -q -p no:cacheprovider tests/test_machine_paths_ratchet.py \
      tests/test_duplicate_helpers_ratchet.py tests/test_check_spec_links.py tests/neuralese/test_line_parity.py
  else
    # Two processes: one pytest over everything grows past 18 GB.
    # Name the active test and dump stacks on a long wait; faulthandler does not
    # terminate a test or change its outcome. See FULL_CHECK_PROGRESS_DECOMPOSITION_20261017.md.
    step python-tests env HF_HOME="$HF_EMPTY" "$PY" -m pytest -v -o faulthandler_timeout=300 -p no:cacheprovider tests --ignore=tests/neuralese
    # The gpu_heavy tests (several GB of CUDA memory each) in a process of their own after the rest, so the suite's peak
    # is the larger of the two rather than their sum (2026-10-12: 13.5 GB in one process).
    step python-neuralese env HF_HOME="$HF_EMPTY" "$PY" -m pytest -v -o faulthandler_timeout=300 -p no:cacheprovider -m 'not gpu_heavy' tests/neuralese
    step python-neuralese-gpu env HF_HOME="$HF_EMPTY" "$PY" -m pytest -v -o faulthandler_timeout=300 -p no:cacheprovider -m gpu_heavy tests/neuralese
  fi
fi
if [ $ts = 1 ]; then
  if [ $quick = 1 ]; then
    step ts-check sh -c 'cd ts-host && npm run --silent check'
  else
    if [ $node_build_done = 0 ]; then
      step ts-build-node sh -c 'cd ts-host && npm run --silent build:node'
    fi
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
