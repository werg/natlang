#!/bin/sh
# Bring the vLLM teacher (container natlang-qwen36-nvfp4-server) up under the ledger if it is not serving, and wait
# until it answers. Several queued campaigns call this; a unit that is already starting is waited for, not duplicated.
ROOT=/home/werg/natlang
UNIT=natlang-qwen36-teacher
# One caller starts it; the others wait on the lock and then for the server.
exec 9>"$HOME/.local/state/natlang/ensure-teacher.lock"
flock 9
if ! curl -sf -m 5 http://127.0.0.1:8082/v1/models >/dev/null; then
  if ! systemctl --user is-active -q $UNIT && ! docker ps --format '{{.Names}}' | grep -qx natlang-qwen36-nvfp4-server; then
    systemctl --user reset-failed $UNIT 2>/dev/null
    python3 $ROOT/scripts/memory_ledger.py run --unit $UNIT --budget-gb 64 --class service --wait 28800 --workdir $ROOT \
      -- docker start -a natlang-qwen36-nvfp4-server || exit 1
  fi
  until curl -sf -m 5 http://127.0.0.1:8082/v1/models >/dev/null; do sleep 15; done
fi
