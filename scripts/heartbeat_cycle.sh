#!/bin/sh
# One heartbeat cycle (plans/HEARTBEAT_PROGRAM.md), through the memory ledger, never overlapping another cycle.
#   HEARTBEAT_REPO     checkout to read and record in (default: the checkout this script lives in)
#   HEARTBEAT_PROFILE  natlang model profile of the executor (default: the default profile)
#   HEARTBEAT_ARGS     extra arguments for `cycle`, for example "--stage shadow"
set -eu
REPO="${HEARTBEAT_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
PATH="$HOME/.local/bin:$PATH"; export PATH
mkdir -p "$REPO/.coordination" "$REPO/runs/heartbeat"
exec 9>"$REPO/.coordination/heartbeat.lock"
if ! flock -n 9; then echo "a heartbeat cycle is already running"; exit 0; fi

UNIT="natlang-heartbeat-cycle"
LOG="$REPO/runs/heartbeat/cycle.log"
# The cycle itself is a client of an already running executor, so its claim is small; "everything that loads a model goes
# through the ledger" holds all the same.
python3 "$REPO/scripts/memory_ledger.py" run --unit "$UNIT" --budget-gb 0.5 --reserve-gb 0 --class collection --wait 900 --workdir "$REPO" -- \
  sh -c "natlang run applications/heartbeat ${HEARTBEAT_PROFILE:+--profile $HEARTBEAT_PROFILE} -- cycle --repo '$REPO' ${HEARTBEAT_ARGS:-} >> '$LOG' 2>&1"
# `run` returns once the unit is launched; hold the lock until it is done.
while systemctl --user is-active --quiet "$UNIT.service"; do sleep 10; done
