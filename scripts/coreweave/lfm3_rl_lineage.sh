#!/bin/bash
# One LFM3 whole-graph RL lineage (JOLT+ or self-distillation) on one CoreWeave GPU sandbox: the actor (natlang SGLang
# fork behind serve_native_checkpoint.py) and the learner (natlang_neuralese.rl.native_stage) alternate in this
# sandbox exactly as scripts/alt_par.sh runs one lineage on DGX: collection, then the update, round after round.
# DGX side: scripts/coreweave/rl_cloud.py (plans/neuralese/LFM3_RL_CLOUD_20261031.md). Job:
# scripts/coreweave/jobs/lfm3/rl-lineage.job.json.
#
#   lfm3_rl_lineage.sh ROOT LINEAGE ROUNDS          (ROUNDS 0 = until stopped)
#
# 1. environments (cached on the volume): the learner runs on the image's Python (NGC PyTorch + the exact wheelhouse)
#    through a --system-site-packages venv at the DGX path the configs name (/home/werg/natlang/.venv-neuralese); the
#    actor's SGLang fork gets its own uv venv at the path training/neuralese/sglang-fork.json names, installed from
#    the x86_64 lock compiled on DGX for the fork commit (public wheels: torch cu130, sglang-kernel, flashinfer) plus
#    the fork's Python source (editable, --no-deps); root node_modules by npm ci (native addons for this arch).
# 2. runtime ROOT/runtime: the bundle's code snapshot + node_modules + the prebuilt ts-host + the held/dev task packets
#    under runs/ that the custody checks read (build_runtime.sh semantics), with its commit in ROOT/runtime/COMMIT.
# 3. parent: a lineage without state.json is materialized from the secret rl-parent (signed GET URLs of the newest
#    DGX-verified export or seed; every byte verified) by rl_cloud_sandbox.py fetch.
# 4. rounds: `natlang_neuralese.rl.production run --rounds 1` on ROOT/LINEAGE/config.json. After every round that
#    advanced trained_rounds, rl_cloud_sandbox.py stage writes an immutable export; the DGX exporter uploads it
#    (signed PUT URLs) and GETs it to the hub, then marks it uploaded here; prune then frees superseded checkpoints.
#    The last round (ROUNDS > 0) waits for its export to be uploaded before the job ends.
set -uo pipefail
ROOT=$1; LINEAGE=$2; ROUNDS=${3:-0}
L=$ROOT/$LINEAGE; E=$ROOT/.exports/$LINEAGE; R=$ROOT/runtime; WORK=${NATLANG_CW_WORK:-/work}
OUT=$ROOT/.sandbox/$LINEAGE; mkdir -p "$OUT" "$E"
exec > >(tee -a "$OUT/driver.log") 2>&1
HELPER="$NATLANG_REPO/scripts/coreweave/rl_cloud_sandbox.py"
log() { echo "{\"at\":\"$(date -u +%FT%TZ)\",\"lineage\":\"$LINEAGE\",$1}"; }
t0=$(date +%s); since() { echo $(( $(date +%s) - $1 )); }
{ echo "== $(date -u +%FT%TZ) sandbox ${CW_SANDBOX_ID:-}"; nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv
  echo "nproc $(nproc)"; echo "memory.max $(cat /sys/fs/cgroup/memory.max 2>/dev/null)"; echo "cpu.max $(cat /sys/fs/cgroup/cpu.max 2>/dev/null)"
  free -g; df -h "$WORK"; python3 --version; } >> "$OUT/host.txt" 2>&1
log "\"event\":\"start\",\"gpu\":\"$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)\",\"nproc\":$(nproc)"

# 1. environments
INPUTS=$ROOT/inputs
VENV=/home/werg/natlang/.venv-neuralese
SGL_DIR=/home/werg/sglang-natlang
SGL_VENV=$(python3 -c "import json;print(json.load(open('$NATLANG_REPO/training/neuralese/sglang-fork.json'))['python'])" | xargs dirname | xargs dirname)
SGL_COMMIT=$(python3 -c "import json;print(json.load(open('$NATLANG_REPO/training/neuralese/sglang-fork.json'))['commit'])")
if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv --system-site-packages "$VENV" || { log '"event":"failed","step":"learner-venv"'; exit 2; }
fi
if [ ! -f "$SGL_VENV/.natlang-$SGL_COMMIT" ]; then
  s=$(date +%s)
  rm -rf "$SGL_DIR" "$SGL_VENV"; mkdir -p "$SGL_DIR"
  tar -xzf "$INPUTS/sglang-natlang-$SGL_COMMIT.tar.gz" -C "$SGL_DIR" || { log '"event":"failed","step":"fork-source"'; exit 2; }
  python3 -m pip install -q --no-cache-dir uv >/dev/null 2>&1 || { log '"event":"failed","step":"uv"'; exit 2; }
  python3 -m uv venv -q -p "$(command -v python3)" "$SGL_VENV" || { log '"event":"failed","step":"sglang-venv"'; exit 2; }
  python3 -m uv pip install -q -p "$SGL_VENV/bin/python" -r "$INPUTS/sglang-lock-$SGL_COMMIT-x86_64.txt" \
    --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match > "$OUT/sglang-install.log" 2>&1 \
    || { tail -20 "$OUT/sglang-install.log"; log '"event":"failed","step":"sglang-deps"'; exit 2; }
  python3 -m uv pip install -q -p "$SGL_VENV/bin/python" --no-deps -e "$SGL_DIR/python" >> "$OUT/sglang-install.log" 2>&1 \
    || { tail -20 "$OUT/sglang-install.log"; log '"event":"failed","step":"sglang-fork"'; exit 2; }
  "$SGL_VENV/bin/python" -c "import sglang, sgl_kernel, torch; print(sglang.__file__, torch.__version__, torch.cuda.is_available())" >> "$OUT/sglang-install.log" 2>&1 \
    || { tail -20 "$OUT/sglang-install.log"; log '"event":"failed","step":"sglang-import"'; exit 2; }
  touch "$SGL_VENV/.natlang-$SGL_COMMIT"
  log "\"event\":\"sglang_ready\",\"commit\":\"$SGL_COMMIT\",\"seconds\":$(since $s)"
fi
COMMIT=$(cat "$NATLANG_REPO/.coreweave-code-commit" 2>/dev/null || python3 -c "import json;print(json.load(open('${NATLANG_CW_CONTROL_DIR}/bundle.json'))['code']['git_sha'])")
if [ "$(cat "$R/COMMIT" 2>/dev/null)" != "$COMMIT" ]; then
  s=$(date +%s)
  (cd "$NATLANG_REPO" && npm ci --no-audit --no-fund > "$OUT/npm-root.log" 2>&1) || { tail -20 "$OUT/npm-root.log"; log '"event":"failed","step":"npm-root"'; exit 2; }
  if [ ! -f "$NATLANG_REPO/ts-host/dist/native/prompt.js" ]; then
    (cd "$NATLANG_REPO/ts-host" && npm ci --no-audit --no-fund && npm run -s build:node) > "$OUT/tshost.log" 2>&1 \
      || { tail -20 "$OUT/tshost.log"; log '"event":"failed","step":"ts-host"'; exit 2; }
  fi
  rm -rf "$R.partial" "$R"; mkdir -p "$R.partial"
  (cd "$NATLANG_REPO" && tar --exclude=./.venv-neuralese --exclude=./.coreweave-prebuilt -cf - .) | tar -x -C "$R.partial"
  cp -an "$INPUTS/runtime-runs/." "$R.partial/runs/"
  echo "$COMMIT" > "$R.partial/COMMIT"; mv "$R.partial" "$R"
  log "\"event\":\"runtime_ready\",\"commit\":\"$COMMIT\",\"seconds\":$(since $s)"
fi
NODE_BIN=$(dirname "$(command -v node)")

# 3. parent
if [ ! -f "$L/state.json" ] && [ ! -f "$L/parent/checkpoint.pt" ]; then
  P="${NATLANG_CW_SECRETS:-$NATLANG_CW_CONTROL_DIR/secrets}/rl-parent"
  [ -f "$P" ] || { log '"event":"failed","step":"no parent capability"'; exit 2; }
  s=$(date +%s)
  python3 "$HELPER" fetch "$P" "$L" || { log '"event":"failed","step":"parent-fetch"'; exit 2; }
  log "\"event\":\"parent_ready\",\"seconds\":$(since $s)"
fi
rm -f "${NATLANG_CW_SECRETS:-$NATLANG_CW_CONTROL_DIR/secrets}/rl-parent"
log "\"event\":\"setup_done\",\"seconds\":$(since $t0)"

# 4. rounds
export NATLANG_REPO=$R PYTHONPATH=$R/training/neuralese:$R PATH=$NODE_BIN:$VENV/bin:$PATH
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 NATLANG_HASH_CACHE=$ROOT/hash-cache.jsonl
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
trained() { python3 -c "import json;print(json.load(open('$L/state.json')).get('trained_rounds',0))" 2>/dev/null || echo 0; }
child=
trap 'log "\"event\":\"term\""; [ -n "$child" ] && kill -TERM $child 2>/dev/null; wait $child 2>/dev/null; exit 143' TERM INT
i=0
rebind() {  # apply_pending_switch.sh semantics: an unbound round takes this sandbox's runtime (actor entry, inventory)
  [ -f "$L/state.json" ] && { r=$(python3 -c "import json;print(json.load(open('$L/state.json'))['round'])")
    ls "$L"/round-$(printf %06d "$r")*/collection-binding.json >/dev/null 2>&1 && return 0; }
  (cd "$R" && "$VENV/bin/python" - "$L/config.json" "$R" <<'PY'
import json, sys
from natlang_neuralese.rl.production import actor_runtime_configuration, bind_actor_runtime_command, validate_actor_runtime_configuration
p, r = sys.argv[1:]
c = json.load(open(p)); before = json.dumps(c, sort_keys=True)
c.update(actor_runtime_configuration(r)); c['actor_command'] = bind_actor_runtime_command(c['actor_command'], r)
c['runtime'] = r + '/ts-host'
validate_actor_runtime_configuration(c)
if json.dumps(c, sort_keys=True) != before:
    json.dump(c, open(p, 'w'), indent=2)
    print(json.dumps({'event': 'actor_rebound', 'runtime': r, 'python_runtime': c['native_actor_python_runtime']['sha256'][:12]}))
PY
  )
}
while [ "$ROUNDS" -eq 0 ] || [ $i -lt "$ROUNDS" ]; do
  rebind || { log '"event":"failed","step":"actor rebind"'; exit 3; }
  before=$(trained); s=$(date +%s)
  log "\"event\":\"round_start\",\"trained_rounds\":$before"
  (cd "$R" && exec "$VENV/bin/python" -u -m natlang_neuralese.rl.production run --config "$L/config.json" --rounds 1) \
    >> "$OUT/controller.log" 2>&1 &
  child=$!; wait $child; rc=$?; child=
  after=$(trained)
  log "\"event\":\"round_done\",\"rc\":$rc,\"trained_rounds\":$after,\"seconds\":$(since $s),\"memory_peak\":\"$(cat /sys/fs/cgroup/memory.peak 2>/dev/null)\""
  if [ $rc -ne 0 ] || [ "$after" -le "$before" ]; then
    tail -30 "$OUT/controller.log"
    log "\"event\":\"EXIT\",\"reason\":\"did not advance\",\"rc\":$rc"; exit 3
  fi
  python3 "$HELPER" stage "$L" "$E" >> "$OUT/exports.log" 2>&1 || log '"event":"stage_failed"'
  python3 "$HELPER" prune "$L" "$E" >> "$OUT/exports.log" 2>&1 || true
  i=$((i+1))
done
# the last staged export must reach the DGX before the job ends
seq=$(ls "$E" | grep -E '^[0-9]{6}\.json$' | tail -1 | cut -c1-6)
log "\"event\":\"await_export\",\"seq\":\"$seq\""
while [ -n "$seq" ] && [ ! -f "$E/$seq.uploaded" ]; do sleep 30; done
log "\"event\":\"done\",\"seconds\":$(since $t0)"
