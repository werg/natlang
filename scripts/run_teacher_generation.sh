#!/usr/bin/env bash
# Coverage-enforced, interruptible teacher generation. Re-running resumes atomic jobs.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
SERVER="${TEACHER_SERVER:-http://127.0.0.1:8081}"
MODEL="${TEACHER_MODEL:-Ternary-Bonsai-2-27B-PTQ1_0}"
SEED="${TEACHER_SEED:-909}"
# Bonsai already saturates this 8 GiB GPU with one decode. Two full agent
# contexts exceed the server's 5 GiB host-memory cgroup on complex programs.
WORKERS="${TEACHER_WORKERS:-1}"
# A typical nested algorithm episode needs 12-20 tool turns.  Keep it in one
# conversation when possible; longer work still checkpoints into durable state.
SEGMENT_TURNS="${TEACHER_SEGMENT_TURNS:-24}"
SEGMENT_MESSAGES="${TEACHER_SEGMENT_MESSAGES:-48}"
SELECTION="${TEACHER_SELECTION:-data/teacher/coverage-selection-s909.ir.jsonl}"
PROGRAM_JOBS="${TEACHER_PROGRAM_JOBS:-runs/teacher-program-balanced-s909-pass3.jobs}"
PROGRAM_OUT="${TEACHER_PROGRAM_OUT:-runs/teacher-program-coverage.ir.jsonl}"
PROGRAM_TURNS="${TEACHER_PROGRAM_TURNS:-data/teacher-program-coverage.turns.jsonl}"
PROGRAM_SFT="${TEACHER_PROGRAM_SFT:-data/teacher-program-coverage.sft.jsonl}"
STUDIO_CASES="${TEACHER_STUDIO_CASES:-data/teacher/studio-cases-v1.jsonl}"
STUDIO_JOBS="${TEACHER_STUDIO_JOBS:-runs/teacher-studio-coverage.jobs}"
STUDIO_TURNS="${TEACHER_STUDIO_TURNS:-data/teacher-studio-coverage.turns.jsonl}"
STUDIO_SFT="${TEACHER_STUDIO_SFT:-data/teacher-studio-coverage.sft.jsonl}"
exec 9>runs/teacher-generation.lock
flock -n 9 || { echo "another teacher generation pipeline holds runs/teacher-generation.lock" >&2; exit 3; }
settle="${TEACHER_SETTLE_SECONDS:-5}"
snapshot() {
  node ts-host/scripts/build-teacher-coverage-selection.mjs "$SELECTION" --seed "$SEED"
  node ts-host/scripts/freeze-studio-teacher-cases.mjs "$STUDIO_CASES" 6
}
fingerprint() {
  sha256sum "$SELECTION" "$SELECTION.manifest.json" "$STUDIO_CASES" \
    "$STUDIO_CASES.manifest.json" training/teacher_coverage.json | sha256sum | cut -d' ' -f1
}

wave=0
while true; do
  wave=$((wave + 1))
  echo "teacher generation wave $wave: discovering current sources"
  snapshot
  # A newly discovered codebase without a provider is a pending coverage gap.
  # Keep rescanning so a concurrently added provider/corpus joins this run.
  while ! node ts-host/scripts/audit-teacher-coverage.mjs \
      --studio-cases "$STUDIO_CASES" --program-ir "$SELECTION"; do
    echo "coverage incomplete; rescanning in ${settle}s" >&2
    sleep "$settle"
    snapshot
  done
  before="$(fingerprint)"
  node ts-host/scripts/teacher-collector.mjs \
    "$SELECTION" "$PROGRAM_JOBS" "$PROGRAM_OUT" \
    --server "$SERVER" --model-id "$MODEL" --root-seed "$SEED" \
    --limit "$(wc -l < "$SELECTION")" --workers "$WORKERS" \
    --segment-turns "$SEGMENT_TURNS" --segment-messages "$SEGMENT_MESSAGES" \
    --cache-stable-tools
  node ts-host/scripts/materialize-native-teacher.mjs \
    "$PROGRAM_OUT" "$PROGRAM_TURNS" --replace
  node ts-host/scripts/export-native-sft.mjs "$PROGRAM_TURNS" "$PROGRAM_SFT" \
    --server "$SERVER" --workers 4
  node ts-host/scripts/collect-studio-teacher.mjs "$STUDIO_CASES" "$STUDIO_JOBS" \
    --server "$SERVER" --model "$MODEL" --seed "$SEED" --workers "$WORKERS"
  node ts-host/scripts/materialize-studio-teacher.mjs \
    "$STUDIO_JOBS" "$STUDIO_TURNS" --cases "$STUDIO_CASES"
  node ts-host/scripts/export-native-sft.mjs "$STUDIO_TURNS" "$STUDIO_SFT" \
    --server "$SERVER" --workers 4
  # Refreeze after all slow work. Any source/corpus that arrived during this
  # wave changes the fingerprint and is processed in the next wave.
  snapshot
  after="$(fingerprint)"
  if [ "$before" = "$after" ]; then
    sleep "$settle"
    snapshot
    [ "$after" = "$(fingerprint)" ] && break
  fi
  echo "sources changed during wave $wave; continuing with the new snapshot"
done
echo "teacher generation converged after $wave wave(s)"
