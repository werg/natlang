#!/usr/bin/env bash
# Restore a missing teacher; never kill a live container because it is loading or busy.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PORT="${1:-8081}"; SLOTS="${2:-${BONSAI_SLOTS:-4}}"; CTX="${3:-53248}"
mkdir -p "$ROOT/runs"
exec 9>"$ROOT/runs/bonsai-watch.lock"
flock -n 9 || exit 1
RETRY_SECONDS=30
while true; do
  if curl --max-time 5 -sf "localhost:$PORT/health" >/dev/null; then
    RETRY_SECONDS=30
  elif ! docker inspect natlang-bonsai >/dev/null 2>&1; then
    echo "$(date -u +%FT%TZ) restoring missing bonsai ($SLOTS slots, context $CTX)" >> "$ROOT/runs/bonsai-watch.log"
    # Closing the lock descriptor in the child keeps the singleton lock owned by this watcher.
    "$ROOT/scripts/serve_bonsai.sh" "$PORT" "$CTX" 99 "$SLOTS" 9>&- >> "$ROOT/runs/bonsai-server.log" 2>&1 &
    sleep "$RETRY_SECONDS"
    RETRY_SECONDS=$((RETRY_SECONDS * 2))
    [ "$RETRY_SECONDS" -le 300 ] || RETRY_SECONDS=300
  else
    echo "$(date -u +%FT%TZ) existing bonsai container not healthy; waiting without replacement" >> "$ROOT/runs/bonsai-watch.log"
  fi
  sleep 20
done
