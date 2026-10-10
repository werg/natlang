#!/bin/sh
# Runs the triage example and stops it (SIGTERM, own PID) as soon as the classify fan-out is finished (first is_urgent trace appears).
export NATLANG_CONFIG_HOME=/tmp/bench/cfg NATLANG_STATE_HOME=/tmp/bench/state-$1 NATLANG_HOME=/tmp/bench/pk NATLANG_CACHE_HOME=/tmp/bench/cache BENCH_KEY=x NATLANG_PROFILE=$1
cd /home/werg/natlang/.claude/worktrees/agent-a3e5fda6488bf57f1
date -u +%FT%T > /tmp/bench/$1.start
scripts/natlang run examples/triage --no-adaptation -- $2 > /tmp/bench/$1.out 2> /tmp/bench/$1.err &
PID=$!
while kill -0 $PID 2>/dev/null; do
  if grep -l '"definition_name":"is_urgent"' /tmp/bench/state-$1/*/triage/traces/*.jsonl >/dev/null 2>&1; then
    date -u +%FT%T > /tmp/bench/$1.classify-done
    sleep 5
    kill $PID
    break
  fi
  sleep 5
done
date -u +%FT%T > /tmp/bench/$1.end
echo done >> /tmp/bench/$1.end
