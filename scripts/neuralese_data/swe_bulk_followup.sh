#!/bin/sh
# Chain after the bulk SWE-rebench records run (harness_bench_bulk.sh, D13): wait until every shard has its records,
# render program-header and whole-trajectory text twins under the ledger (low priority, resumable), then register and publish the registry entries
# and manifests (pushed by register_corpora.py), and brief Pop. Costs nothing while it waits; safe to rerun (each step resumes).
#   nohup scripts/neuralese_data/swe_bulk_followup.sh BULK_OUT TWIN_ROOT > LOG 2>&1 &
set -eu
BULK=$1; TWINS=$2
ROOT=/home/werg/natlang
LFM=/home/werg/.cache/huggingface/hub/models--LiquidAI--LFM2.5-350M/snapshots/9e6c6ccf47cd318696e137d381a7ded8fe4df09f
MELLUM=/home/werg/data/models/mellum21-12b-a2.5b-thinking
cd "$ROOT"
all_records() {
  [ -e "$BULK/shards/.split" ] || return 1
  for d in "$BULK"/shards/[0-9]*; do [ -e "$d/summary.json" ] || return 1; done
}
until all_records; do sleep 300; done
echo "records complete $(date -u +%FT%TZ)"
mkdir -p "$TWINS"
all_twins() {
  for name in lfm25 mellum21; do
    [ -e "$TWINS/$name/receipt.json" ] || return 1
    [ -e "$TWINS/program-header/$name/receipt.json" ] || return 1
  done
}
until all_twins; do
  UNIT=swe-norm-twins-$(date +%s)
  python3 scripts/memory_ledger.py run --unit "$UNIT" --budget-gb 12 --host-max-gb 24 --priority low --resumable \
    --class collection --wait 86400 --workdir "$ROOT" -- \
    .venv-neuralese/bin/python scripts/neuralese_data/swe_bulk_twins.py --bulk "$BULK" --twin-root "$TWINS" \
    --tokenizer lfm25="$LFM" --tokenizer mellum21="$MELLUM" --parallel 1
  # The ledger may stop and requeue the unit; wait for it to finish one way or the other, then check.
  while systemctl --user is-active -q "$UNIT.service" || python3 scripts/memory_ledger.py status | grep -q "\"$UNIT.service\""; do sleep 120; done
  all_twins || sleep 600
done
echo "twins complete $(date -u +%FT%TZ)"
# Registration commits and pushes from a private worktree off origin/main (register_corpora.py), never from the
# shared checkout; it retries races and red checks itself.
.venv-neuralese/bin/python scripts/neuralese_data/register_swe_bulk.py --bulk "$BULK" --twin-root "$TWINS" || { echo "register exit $?"; exit 1; }
python3 scripts/coord.py --as dgx-datagen-swe send --to pop --subject "D13 done: records + four LFM/Mellum text twins registered" \
  -m "Registered harness-bench-swe-rebench-openhands-pi-records-bulk-20261011-v1 and program-header plus whole-trajectory twins for LFM/Mellum. Whole trajectories retain the final clean assistant target. Training admission is recorded per registry entry; additive to v4 (parquet rows >= 1000). Counts and exact renderer/tokenizer hashes are in registry receipts. Built by training/neuralese/scripts/harness_bench_bulk.sh + scripts/neuralese_data/swe_bulk_twins.py." || true
echo "done $(date -u +%FT%TZ)"
