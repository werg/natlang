#!/bin/bash
# Bounded LFM3 joint-SFT shakedown on one CoreWeave GPU sandbox (LFM3_COMMITTED_DECISIONS_20261028.md #6;
# plans/neuralese/LFM3_SFT_RUN_20261028.md). Runs the ordinary native trainer, interrupts it with SIGTERM after its
# first rolling checkpoint (the trainer saves at the signal), restarts the identical command so it restores
# weights/optimizer/RNG/exposure from its own rolling state, and lets it finish. Records host RAM (cgroup) and GPU
# memory every 5 s, so the learner RAM minimum and DiLoCo local_steps come from measurement.
#
#   lfm3_sft_check.sh OUT ARGS.json [STOP_AFTER_CHECKPOINT_STEP]
#
# ARGS.json is the trainer argument list (a JSON array, without --out; OUT/train is appended).
set -uo pipefail
OUT=$1; ARGS=$2; STOP_AT=${3:-100}
mkdir -p "$OUT"
facts() {
  { echo "== at $(date -u +%FT%TZ)"; nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu,driver_version,compute_cap --format=csv
    echo "nproc $(nproc)"; echo "cgroup memory.max $(cat /sys/fs/cgroup/memory.max 2>/dev/null)"; free -g
    lscpu | grep -E 'Model name|^CPU\(s\)'; df -h "${NATLANG_CW_WORK:-/work}"; } >> "$OUT/host.txt" 2>&1
}
facts
( while true; do
    gpu=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    echo "{\"t\":$(date +%s),\"cg_current\":$(cat /sys/fs/cgroup/memory.current 2>/dev/null || echo null),\"cg_peak\":$(cat /sys/fs/cgroup/memory.peak 2>/dev/null || echo null),\"gpu_mib_util\":\"$gpu\",\"load\":\"$(cut -d' ' -f1-3 /proc/loadavg)\"}"
    sleep 5
  done ) >> "$OUT/resources.jsonl" &
sampler=$!
pid=''
cleanup() {
  if [ -n "$pid" ]; then kill -TERM "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true; fi
  kill "$sampler" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 143' TERM INT
# The pinned base comes from the hub at its exact revision (faster than pushing 3.4 GB from DGX); "@@BASE@@" in
# ARGS.json is replaced by the snapshot directory (the trainer records and checks its base inventory).
BASE=$(python3 - <<'PY'
import hashlib, os, sys
from huggingface_hub import snapshot_download
sys.path.insert(0, 'training/neuralese')
from natlang_neuralese.common.line_backbones import backbone
line = backbone('lfm3')
path = snapshot_download(line['hf_repo'], revision=line['revision'], cache_dir=os.environ.get('HF_HUB_CACHE', '/work/hf'))
print(path)
PY
) || { echo "base download failed" >> "$OUT/phases.log"; exit 1; }
echo "$(date -u +%FT%TZ) base $BASE" >> "$OUT/phases.log"
mapfile -t TRAINER < <(python3 -c 'import json,sys; [print(a.replace("@@BASE@@", sys.argv[2])) for a in json.load(open(sys.argv[1]))]' "$ARGS" "$BASE")
CMD=(python -m natlang_neuralese.train.native_state_sft "${TRAINER[@]}" --out "$OUT/train")
echo "$(date -u +%FT%TZ) phase A start" >> "$OUT/phases.log"
"${CMD[@]}" > "$OUT/train-a.log" 2>&1 &
pid=$!
while kill -0 $pid 2>/dev/null; do
  if python3 - "$OUT/train/progress.json" "$STOP_AT" <<'PY' 2>/dev/null
import json, sys
p = json.load(open(sys.argv[1]))
sys.exit(0 if (p.get('checkpoint_step') or 0) >= int(sys.argv[2]) else 1)
PY
  then
    echo "$(date -u +%FT%TZ) SIGTERM after checkpoint $(python3 -c "import json;print(json.load(open('$OUT/train/progress.json'))['checkpoint_step'])")" >> "$OUT/phases.log"
    kill -TERM $pid; break
  fi
  sleep 5
done
wait "$pid"
rc=$?
pid=''
echo "$(date -u +%FT%TZ) phase A exit $rc" >> "$OUT/phases.log"
if [ "$rc" -ne 0 ]; then
  echo "$(date -u +%FT%TZ) phase A failed; no successful seed or phase B" >> "$OUT/phases.log"
  exit "$rc"
fi
# A zero process status without a committed full checkpoint cannot seed distributed continuation.
if [ ! -s "$OUT/train/checkpoint.pt" ] || ! python3 - "$OUT/train/progress.json" <<'PY_CHECK'
import json, sys
p = json.load(open(sys.argv[1]))
sys.exit(0 if p.get('checkpoint_step', 0) > 0 else 1)
PY_CHECK
then
  echo "$(date -u +%FT%TZ) phase A produced no committed checkpoint" >> "$OUT/phases.log"
  exit 1
fi
cp "$OUT/train/progress.json" "$OUT/progress-a.json" || exit 1
facts
[ "${CHECK_PHASE_B:-1}" = 0 ] && { echo "$(date -u +%FT%TZ) phase B skipped (CHECK_PHASE_B=0: seed run)" >> "$OUT/phases.log"; exit 0; }
echo "$(date -u +%FT%TZ) phase B start (same command: restore from the rolling state)" >> "$OUT/phases.log"
"${CMD[@]}" > "$OUT/train-b.log" 2>&1
rc=$?
echo "$(date -u +%FT%TZ) phase B exit $rc" >> "$OUT/phases.log"
facts
exit $rc
