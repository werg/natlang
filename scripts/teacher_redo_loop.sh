#!/usr/bin/env bash
# A second teacher redoes what the first teacher's runs were refused for, while the first keeps collecting.
# Usage: scripts/teacher_redo_loop.sh SHARD.ir.jsonl FIRST.results.jsonl OUT_PREFIX [--while-pid PID] -- COLLECTOR_ARGS...
#   e.g. scripts/teacher_redo_loop.sh data/.../shard-train-v3.ir.jsonl runs/inline-curriculum/train3.results.jsonl \
#          runs/inline-curriculum/train3-luna --while-pid 1234 -- --provider openai-codex --model-id gpt-6-luna \
#          --root-seed 909 --workers 1 --context-tokens 16384 --max-turns 20 --reasoning-effort low --execution-plans
# Each round admits the first teacher's results (as build_lora_sft.sh does, --require-technique), appends newly refused
# programs to OUT_PREFIX.ir.jsonl, and runs the collector over it into OUT_PREFIX.jobs / OUT_PREFIX.results.jsonl
# (finished jobs are kept, so only new programs run). While PID (the first teacher's collector) lives, rounds repeat
# every NATLANG_REDO_INTERVAL seconds (default 1200); after it exits, one last round runs. NATLANG_TS_HOST picks the
# ts-host whose dist runs the collector (default: this checkout's).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SHARD="$(realpath "${1:?shard}")"; FIRST="$(realpath -m "${2:?first teacher results}")"; OUT="${3:?output prefix}"; shift 3
OUT="$(realpath -m "$OUT")"
WHILE_PID=""
if [ "${1:-}" = "--while-pid" ]; then WHILE_PID="$2"; shift 2; fi
[ "${1:-}" = "--" ] && shift
TS_HOST="${NATLANG_TS_HOST:-$ROOT/ts-host}"
INTERVAL="${NATLANG_REDO_INTERVAL:-1200}"
round() {
  # The export covers only the selected range. Include completed jobs from earlier ranges too.
  # Override for collectors whose jobs directory does not use the conventional sibling name.
  local source="$FIRST"
  local jobs="${NATLANG_REDO_JOBS:-${FIRST%.results.jsonl}.jobs}"
  if [ -d "$jobs" ]; then
    source="$OUT.first-snapshot.jsonl"
    node "$ROOT/scripts/snapshot_teacher_jobs.mjs" "$jobs" "$source"
  fi
  node --max-old-space-size=3000 "$ROOT/ts-host/scripts/inline-curriculum/admit.mjs" "$source" \
    --ledger "$OUT.first-ledger.jsonl" --require-technique | tail -3 | head -1
  node "$ROOT/ts-host/scripts/inline-curriculum/redo.mjs" "$SHARD" --ledger "$OUT.first-ledger.jsonl" --out "$OUT.ir.jsonl"
  [ -s "$OUT.ir.jsonl" ] || return 0
  (cd "$TS_HOST" && node dist/teacher/cli.js "$OUT.ir.jsonl" "$OUT.jobs" "$OUT.results.jsonl" --all "$@")
}
for ((;;)); do
  alive=0; [ -n "$WHILE_PID" ] && kill -0 "$WHILE_PID" 2>/dev/null && alive=1
  echo "[$(date +%FT%T)] round (first teacher running: $alive)"
  round "$@"
  [ "$alive" = 1 ] || break
  sleep "$INTERVAL"
done
