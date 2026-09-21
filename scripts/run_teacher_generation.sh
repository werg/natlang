#!/usr/bin/env bash
# Coverage-enforced, interruptible teacher generation. Re-running resumes atomic jobs.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
SERVER="${TEACHER_SERVER:-http://127.0.0.1:8081}"
MODEL="${TEACHER_MODEL:-Ternary-Bonsai-2-27B-PTQ1_0}"
SEED="${TEACHER_SEED:-909}"
WORKERS="${TEACHER_WORKERS:-2}"
SEGMENT_TURNS="${TEACHER_SEGMENT_TURNS:-12}"
SEGMENT_MESSAGES="${TEACHER_SEGMENT_MESSAGES:-24}"
SELECTION="${TEACHER_SELECTION:-data/teacher/coverage-selection-s909.ir.jsonl}"
PROGRAM_JOBS="${TEACHER_PROGRAM_JOBS:-runs/teacher-program-balanced-s909-pass3.jobs}"
PROGRAM_OUT="${TEACHER_PROGRAM_OUT:-runs/teacher-program-coverage.ir.jsonl}"
STUDIO_CASES="${TEACHER_STUDIO_CASES:-data/teacher/studio-cases-v1.jsonl}"
STUDIO_JOBS="${TEACHER_STUDIO_JOBS:-runs/teacher-studio-coverage.jobs}"
STUDIO_TURNS="${TEACHER_STUDIO_TURNS:-data/teacher-studio-coverage.turns.jsonl}"
exec 9>runs/teacher-generation.lock
flock -n 9 || { echo "another teacher generation pipeline holds runs/teacher-generation.lock" >&2; exit 3; }
IMPORTS=()
for old in runs/teacher-program-balanced-s909-pass2.ir.jsonl \
           runs/teacher-program-balanced-s909-pass3.ir.jsonl; do
  [ ! -f "$old" ] || IMPORTS+=(--import-ir "$old")
done

settle="${TEACHER_SETTLE_SECONDS:-5}"
snapshot() {
  python scripts/build_teacher_coverage_selection.py "$SELECTION" --seed "$SEED"
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
  while ! python scripts/audit_teacher_coverage.py \
      --studio-cases "$STUDIO_CASES" --program-ir "$SELECTION"; do
    echo "coverage incomplete; rescanning in ${settle}s" >&2
    sleep "$settle"
    snapshot
  done
  before="$(fingerprint)"
  PYTHONPATH=. .venv/bin/python scripts/collect_teacher_batch.py \
    "$SELECTION" "$PROGRAM_JOBS" "$PROGRAM_OUT" \
    --server "$SERVER" --model-id "$MODEL" --root-seed "$SEED" \
    --limit "$(wc -l < "$SELECTION")" --workers "$WORKERS" \
    --segment-turns "$SEGMENT_TURNS" --segment-messages "$SEGMENT_MESSAGES" \
    --cache-stable-tools \
    "${IMPORTS[@]}"
  node ts-host/scripts/collect-studio-teacher.mjs "$STUDIO_CASES" "$STUDIO_JOBS" \
    --server "$SERVER" --model "$MODEL" --seed "$SEED" --workers "$WORKERS"
  PYTHONPATH=. .venv/bin/python scripts/materialize_studio_teacher.py \
    "$STUDIO_JOBS" "$STUDIO_TURNS" --cases "$STUDIO_CASES" --replace
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
