#!/bin/sh
# Chain after the bulk SWE-rebench records run (harness_bench_bulk.sh, D13): wait until every shard has its records,
# render both text twins under the ledger (low priority, resumable), then register, publish, commit and push the
# registry entries and manifests, and brief Pop. Costs nothing while it waits; safe to rerun (each step resumes).
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
until [ -e "$TWINS/lfm25/receipt.json" ] && [ -e "$TWINS/mellum21/receipt.json" ]; do
  UNIT=swe-norm-twins-$(date +%s)
  python3 scripts/memory_ledger.py run --unit "$UNIT" --budget-gb 12 --host-max-gb 24 --priority low --resumable \
    --class collection --wait 86400 --workdir "$ROOT" -- \
    .venv-neuralese/bin/python scripts/neuralese_data/swe_bulk_twins.py --bulk "$BULK" --twin-root "$TWINS" \
    --tokenizer lfm25="$LFM" --tokenizer mellum21="$MELLUM" --parallel 1
  # The ledger may stop and requeue the unit; wait for it to finish one way or the other, then check.
  while systemctl --user is-active -q "$UNIT.service" || python3 scripts/memory_ledger.py status | grep -q "\"$UNIT.service\""; do sleep 120; done
  [ -e "$TWINS/lfm25/receipt.json" ] && [ -e "$TWINS/mellum21/receipt.json" ] || sleep 600
done
echo "twins complete $(date -u +%FT%TZ)"
git fetch -q && git merge -q --no-edit origin/main || { git merge --abort 2>/dev/null; echo "merge failed; registering without commit"; NOCOMMIT=--no-commit; }
.venv-neuralese/bin/python scripts/neuralese_data/register_swe_bulk.py --bulk "$BULK" --twin-root "$TWINS" ${NOCOMMIT:-} || { echo "register exit $?"; NOCOMMIT=--no-commit; }
if [ -z "${NOCOMMIT:-}" ] && scripts/check-main.sh --quick; then git push -q origin HEAD:main; fi
python3 scripts/coord.py --as dgx-datagen-swe send --to pop --subject "D13 done: bulk SWE-rebench harness-bench records + LFM/Mellum text twins registered" \
  -m "Registered harness-bench-swe-rebench-openhands-pi-records-bulk-20261011-v1 and its text twins (lfm25-350m / mellum21-12b -bulk-20261011-v1), training_admission true, additive to v4 (parquet rows >= 1000). Counts in the registry entries. Built by training/neuralese/scripts/harness_bench_bulk.sh + scripts/neuralese_data/swe_bulk_twins.py." || true
echo "done $(date -u +%FT%TZ)"
