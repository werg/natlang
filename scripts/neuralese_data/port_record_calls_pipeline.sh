#!/bin/bash
# Sequential D15/D16 pipeline: each stage runs port_record_calls.sh as its own ledger unit; a stage is published
# (register_corpora.py: manifest committed and pushed from a private worktree, no hand commit) only after its unit
# ended with Result=success and is neither active nor requeued by the ledger: a preempted unit also stops cleanly
# (Result=success) and is relaunched from the ledger queue, so success alone is not completion (2026-10-11: a
# partial S3 snapshot was published that way). A failed stage is relaunched (it resumes) up to TRIES times.
# Run as a light waiter (it claims no memory itself):
#   systemd-run --user --unit s1-natlang-pipeline-v2 --working-directory=/home/werg/natlang \
#     scripts/neuralese_data/port_record_calls_pipeline.sh
set -uo pipefail
cd /home/werg/natlang
DATA=/home/werg/data/natlang-corpora
S3=/home/werg/data/neuralese-s3-full/records
TRIES=${TRIES:-6}; BUDGET=${BUDGET:-8}; WORKERS=${WORKERS:-4}
queued() { python3 scripts/memory_ledger.py status 2>/dev/null | grep -q "\"$1.service\""; }
stage() { # unit corpus-id input-dir corpus-of-records build-args [skip-done-in]
  local unit=$1 id=$2 in=$3 src=$4 args=$5 skip=${6:-} try=0
  [ -f "$DATA/$id/published" ] && return 0
  while [ $try -lt $TRIES ]; do
    # A unit already running or queued (this waiter was restarted) is waited for, not launched twice.
    if ! systemctl --user is-active -q "$unit" && ! queued "$unit"; then
      try=$((try + 1))
      systemctl --user reset-failed "$unit" 2>/dev/null
      python3 scripts/memory_ledger.py run --unit "$unit" --budget-gb "$BUDGET" --priority medium --resumable \
        --wait 28800 --workdir /home/werg/natlang -- \
        env BUILD_ARGS="$args" SKIP_DONE_IN="$skip" scripts/neuralese_data/port_record_calls.sh "$in" "$DATA/$id" "$WORKERS" "$src" || { sleep 600; continue; }
      sleep 30
    fi
    # Running, or stopped by the ledger for a higher priority job and waiting in its queue to be relaunched.
    while systemctl --user is-active -q "$unit" || queued "$unit"; do sleep 120; done
    result=$(systemctl --user show -p Result --value "$unit")
    echo "$(date -Is) $unit attempt $try: $result"
    if [ "$result" = success ]; then
      # An entry not yet on main is registered from the entry file the cutover wrote beside the data.
      reg=(--id "$id"); [ -f "$DATA/$id/registry-entry.json" ] && reg=(--entries "$DATA/$id/registry-entry.json")
      python3 scripts/neuralese_data/register_corpora.py "${reg[@]}" -m "Publish $id (D15/D16 stage finished)" \
        > "$DATA/$id/publish.log" 2>&1 && touch "$DATA/$id/published" && return 0
      echo "$(date -Is) $id: registration failed (see $DATA/$id/publish.log)" >&2
      return 1
    fi
    sleep 300
  done
  echo "$(date -Is) $unit gave up after $TRIES attempts" >&2
}
# Runtime cutover 2026-10-12 (ts-host dist rebuilt with the context-derived page size, 8267b177): v1 stages are frozen
# (S3 v1 keeps the 207 shards replayed at fixed 2,000-character pages; view-ask and s3-large v1 never replayed and are
# superseded). v2 continues S3 with exactly the shards v1 lacks (same IR, SKIP_DONE_IN), and runs the other two.
stage s1-natlang-s3-v2 port-record-calls-s3-20261011-v2 "$S3" s1-full-final-20261003 "" "$DATA/port-record-calls-s3-20261011-v1"
stage s1-natlang-viewask-v2 port-record-calls-view-ask-20261011-v2 data/neuralese/corpora/view-ask-20261010-v4 view-ask-20261010-v4 ""
stage s1-natlang-s3-large-v2 port-record-calls-s3-large-20261011-v2 "$S3" s1-full-final-20261003 "--min-evidence 32000 --max-evidence 120000"
