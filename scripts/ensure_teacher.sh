#!/bin/sh
# Bring the vLLM teacher (container natlang-qwen36-nvfp4-server) up under the ledger if it is not serving, and wait
# until it answers. Several queued campaigns call this; a unit that is already starting is waited for, not duplicated.
# vLLM sizes its KV cache from the memory free while it loads (~10 min): a start that overlaps another big job fails
# with negative KV memory (2026-10-07), so a failed start is retried after a pause, up to ATTEMPTS times. Startup
# (CUDA graph capture) peaks at ~74 GB, serving settles at ~59 GB: 68 GB keeps the peak under the guard's 1.15x.
ROOT=/home/werg/natlang
UNIT=natlang-qwen36-teacher
ATTEMPTS=${ATTEMPTS:-4}
serving() { curl -sf -m 5 http://127.0.0.1:8082/v1/models >/dev/null; }
running() { systemctl --user is-active -q $UNIT || docker ps --format '{{.Names}}' | grep -qx natlang-qwen36-nvfp4-server; }
# One caller starts it; the others wait on the lock and then for the server.
exec 9>"$HOME/.local/state/natlang/ensure-teacher.lock"
flock 9
attempt=0
until serving; do
  if ! running; then
    [ $attempt -ge $ATTEMPTS ] && { echo "teacher failed to start $attempt times" >&2; exit 1; }
    [ $attempt -gt 0 ] && sleep 300
    attempt=$((attempt + 1))
    systemctl --user reset-failed $UNIT 2>/dev/null
    python3 $ROOT/scripts/memory_ledger.py run --unit $UNIT --budget-gb 68 --class service --wait 28800 --workdir $ROOT \
      -- docker start -a natlang-qwen36-nvfp4-server || exit 1
    sleep 30
  fi
  sleep 15
done
