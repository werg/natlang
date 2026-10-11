#!/bin/bash
# Sequential D15/D16 pipeline: each stage runs port_record_calls.sh as its own ledger unit; a stage is published
# (sync_training_corpora.py publish, manifest written uncommitted) only after its unit ends with Result=success, i.e.
# every shard replayed. A failed or preempted stage is relaunched (it resumes) up to TRIES times.
# Run as a light waiter (it claims no memory itself):
#   systemd-run --user --unit s1-natlang-pipeline-v1 --working-directory=/home/werg/natlang \
#     scripts/neuralese_data/port_record_calls_pipeline.sh
set -uo pipefail
cd /home/werg/natlang
DATA=/home/werg/data/natlang-corpora
S3=/home/werg/data/neuralese-s3-full/records
TRIES=${TRIES:-6}; BUDGET=${BUDGET:-8}; WORKERS=${WORKERS:-4}
stage() { # unit corpus-id input-dir corpus-of-records build-args
  local unit=$1 id=$2 in=$3 src=$4 args=$5 try=0
  [ -f "$DATA/$id/published" ] && return 0
  while [ $try -lt $TRIES ]; do
    try=$((try + 1))
    systemctl --user reset-failed "$unit" 2>/dev/null
    python3 scripts/memory_ledger.py run --unit "$unit" --budget-gb "$BUDGET" --wait 28800 --workdir /home/werg/natlang -- \
      env BUILD_ARGS="$args" scripts/neuralese_data/port_record_calls.sh "$in" "$DATA/$id" "$WORKERS" "$src" || { sleep 600; continue; }
    sleep 30
    while systemctl --user is-active -q "$unit"; do sleep 120; done
    result=$(systemctl --user show -p Result --value "$unit")
    echo "$(date -Is) $unit attempt $try: $result"
    if [ "$result" = success ]; then
      python3 scripts/sync_training_corpora.py publish --machine dgx --id "$id" > "$DATA/$id/publish.log" 2>&1 && touch "$DATA/$id/published"
      return 0
    fi
    sleep 300
  done
  echo "$(date -Is) $unit gave up after $TRIES attempts" >&2
}
stage s1-natlang-s3-v1 port-record-calls-s3-20261011-v1 "$S3" s1-full-final-20261003 ""
stage s1-natlang-viewask-v1 port-record-calls-view-ask-20261011-v1 data/neuralese/corpora/view-ask-20261010-v4 view-ask-20261010-v4 ""
stage s1-natlang-s3-large-v1 port-record-calls-s3-large-20261011-v1 "$S3" s1-full-final-20261003 "--min-evidence 32000 --max-evidence 120000"
