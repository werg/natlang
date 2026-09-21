#!/usr/bin/env bash
# Coverage-enforced, interruptible teacher generation. Re-running resumes atomic jobs.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
SERVER="${TEACHER_SERVER:-http://127.0.0.1:8081}"
MODEL="${TEACHER_MODEL:-Ternary-Bonsai-2-27B-PTQ1_0}"
SEED="${TEACHER_SEED:-909}"
WORKERS="${TEACHER_WORKERS:-2}"
SELECTION="${TEACHER_SELECTION:-data/external_pilot/teacher-selection-balanced-s909.ir.jsonl}"
PROGRAM_JOBS="${TEACHER_PROGRAM_JOBS:-runs/teacher-program-balanced-s909-pass3.jobs}"
PROGRAM_OUT="${TEACHER_PROGRAM_OUT:-runs/teacher-program-coverage.ir.jsonl}"
STUDIO_CASES="${TEACHER_STUDIO_CASES:-data/teacher/studio-cases-v1.jsonl}"
STUDIO_JOBS="${TEACHER_STUDIO_JOBS:-runs/teacher-studio-coverage.jobs}"
STUDIO_TURNS="${TEACHER_STUDIO_TURNS:-data/teacher-studio-coverage.turns.jsonl}"
exec 9>runs/teacher-generation.lock
flock -n 9 || { echo "another teacher generation pipeline holds runs/teacher-generation.lock" >&2; exit 3; }

node ts-host/scripts/freeze-studio-teacher-cases.mjs "$STUDIO_CASES" 6
python scripts/audit_teacher_coverage.py --studio-cases "$STUDIO_CASES" --program-ir "$SELECTION"
PYTHONPATH=. .venv/bin/python scripts/collect_teacher_batch.py \
  "$SELECTION" "$PROGRAM_JOBS" "$PROGRAM_OUT" \
  --server "$SERVER" --model-id "$MODEL" --root-seed "$SEED" \
  --limit "$(wc -l < "$SELECTION")" --workers "$WORKERS" \
  --import-ir runs/teacher-program-balanced-s909-pass2.ir.jsonl
node ts-host/scripts/collect-studio-teacher.mjs "$STUDIO_CASES" "$STUDIO_JOBS" \
  --server "$SERVER" --model "$MODEL" --seed "$SEED" --workers "$WORKERS"
PYTHONPATH=. .venv/bin/python scripts/materialize_studio_teacher.py \
  "$STUDIO_JOBS" "$STUDIO_TURNS" --replace
