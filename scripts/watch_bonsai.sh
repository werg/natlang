#!/usr/bin/env bash
# Keep the Bonsai teacher up during long generation jobs: restart it whenever the health check fails.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; PORT="${1:-8081}"; SLOTS="${2:-${BONSAI_SLOTS:-6}}"
while true; do
  if ! curl -sf "localhost:$PORT/health" >/dev/null; then
    echo "$(date +%T) restarting bonsai" >> "$ROOT/runs/bonsai-watch.log"
    "$ROOT/scripts/serve_bonsai.sh" "$PORT" "" 99 "$SLOTS" >> "$ROOT/runs/bonsai-server.log" 2>&1 &
    sleep 120
  fi
  sleep 20
done
