#!/bin/bash
# One LFM3 lineage on one CoreWeave GPU sandbox, driven exactly as the DGX drives it: either a whole-graph RL lineage
# (JOLT+ or self-distillation; actor and learner alternate, `natlang_neuralese.rl.production run --rounds 1` per round)
# or a joint-SFT lineage of windows (each window one mandatory supervised controller that trains to its qualified
# semantic-refresh boundary; the next window continues from that earned checkpoint, scripts/neuralese_data/sft_window.py).
# DGX side: scripts/coreweave/rl_cloud.py (plans/neuralese/LFM3_RL_CLOUD_20261031.md). Job:
# scripts/coreweave/jobs/lfm3/rl-lineage.job.json.
#
#   lfm3_rl_lineage.sh ROOT LINEAGE ROUNDS          (ROUNDS: RL rounds or SFT windows; 0 = until stopped)
#
# 1. parent and inputs: a lineage without a controller state is materialized from the secret rl-parent (signed GET
#    URLs of the newest DGX-verified export or seed, plus the run inputs: a frozen runtime, the exact data closure,
#    the task source; every byte verified, ZIP-delta checkpoints reconstructed over their full bases) by
#    rl_cloud_sandbox.py fetch.
# 2. environments (cached on the volume): the learner runs on the image's Python (NGC PyTorch + the exact wheelhouse)
#    through a --system-site-packages venv at the DGX path the configs name (/home/werg/natlang/.venv-neuralese). Only an
#    actor that uses the SGLang fork (`--engine sglang` in an RL lineage's actor_command) gets the fork's uv venv
#    (x86_64 lock + editable fork source + Rust radix-tree core); native actors and SFT windows need none of it.
# 3. runtime: a lineage whose config names a frozen runtime runs it unchanged (restored by fetch at its DGX path). A
#    lineage bound to ROOT/runtime gets it built from the bundle's code snapshot (build_runtime.sh semantics) and its
#    unbound rounds rebound to it.
# 4. progress: an RL call progresses when trained_rounds advances; an SFT window when it reaches its semantic-refresh
#    boundary with optimizer updates. After every progress rl_cloud_sandbox.py stage writes an immutable export (the
#    DGX exporter uploads it, GETs it to the hub and marks it uploaded; prune then frees superseded checkpoints); SFT
#    windows are also staged every NATLANG_CW_STAGE_SECONDS while they train (an interrupted window resumes in its own
#    rolling slot). The last unit (ROUNDS > 0) waits for its export to be uploaded before the job ends.
set -uo pipefail
ROOT=$1; LINEAGE=$2; ROUNDS=${3:-0}
L=$ROOT/$LINEAGE; E=$ROOT/.exports/$LINEAGE; WORK=${NATLANG_CW_WORK:-/work}
OUT=$ROOT/.sandbox/$LINEAGE; mkdir -p "$OUT" "$E"
exec > >(tee -a "$OUT/driver.log") 2>&1
CODE=$NATLANG_REPO
HELPER="$CODE/scripts/coreweave/rl_cloud_sandbox.py"
WINDOW="$CODE/scripts/neuralese_data/sft_window.py"
# A failed driver keeps the sandbox readable for NATLANG_CW_HOLD_ON_FAILURE seconds (logs, files) before it completes.
trap 'rc=$?; if [ $rc -ne 0 ] && [ $rc -ne 143 ] && [ $rc -ne 75 ]; then sleep "${NATLANG_CW_HOLD_ON_FAILURE:-0}"; fi' EXIT
log() { echo "{\"at\":\"$(date -u +%FT%TZ)\",\"lineage\":\"$LINEAGE\",$1}"; }
t0=$(date +%s); since() { echo $(( $(date +%s) - $1 )); }
{ echo "== $(date -u +%FT%TZ) sandbox ${CW_SANDBOX_ID:-}"; nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv
  echo "nproc $(nproc)"; echo "memory.max $(cat /sys/fs/cgroup/memory.max 2>/dev/null)"; echo "cpu.max $(cat /sys/fs/cgroup/cpu.max 2>/dev/null)"
  free -g; df -h "$WORK"; python3 --version; } >> "$OUT/host.txt" 2>&1
log "\"event\":\"start\",\"gpu\":\"$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)\",\"nproc\":$(nproc)"
# The launch may have backed off to a smaller CPU/RAM shape (cwctl.create_with_shape_backoff): refuse to start work on
# an assigned cgroup below the job's measured minimum (NATLANG_CW_MIN_CPU / NATLANG_CW_MIN_MEMORY), without the
# failure hold, so the sandbox ends at once and the launcher can try another account.
if ! python3 - <<'PY'
import os, sys
def gib(v):
    for s, f in (('Gi', 1), ('Mi', 1 / 1024), ('Ti', 1024), ('G', 1e9 / 2**30), ('M', 1e6 / 2**30)):
        if v.endswith(s): return float(v[:-len(s)]) * f
    return float(v) / 2**30
want_cpu, want_mem = os.environ.get('NATLANG_CW_MIN_CPU', ''), os.environ.get('NATLANG_CW_MIN_MEMORY', '')
try: quota, period = open('/sys/fs/cgroup/cpu.max').read().split()
except OSError: quota, period = 'max', '1'
cpus = os.cpu_count() if quota == 'max' else int(quota) / int(period)
try: raw = open('/sys/fs/cgroup/memory.max').read().strip()
except OSError: raw = 'max'
mem = float('inf') if raw == 'max' else int(raw) / 2**30
low = (want_cpu and cpus < float(want_cpu)) or (want_mem and mem < gib(want_mem) * 0.97)
print(f'{{"assigned_cpu":{cpus},"assigned_memory_gib":{"null" if mem == float("inf") else round(mem, 1)},"min_cpu":"{want_cpu}","min_memory":"{want_mem}","below_minimum":{str(bool(low)).lower()}}}')
sys.exit(3 if low else 0)
PY
then log '"event":"refused","step":"assigned-shape-below-minimum"'; exit 75; fi
progress() { python3 "$HELPER" progress "$L"; }
field() { python3 -c "import json,sys;v=json.loads(sys.argv[1]).get(sys.argv[2]);print('' if v is None else str(v).lower() if isinstance(v,bool) else v)" "$1" "$2"; }

INPUTS=$ROOT/inputs
# 1. parent and run inputs
SECRETS=${NATLANG_CW_SECRETS:-$NATLANG_CW_CONTROL_DIR/secrets}
if [ ! -f "$L/config.json" ] && [ ! -f "$L/lineage.json" ]; then
  P="$SECRETS/rl-parent"
  [ -f "$P" ] || log '"event":"await_parent","note":"rl_cloud.py deliver-parent writes the capability"'
  while [ ! -f "$P" ]; do sleep 20; done
  s=$(date +%s)
  python3 "$HELPER" fetch "$P" "$L" || { log '"event":"failed","step":"parent-fetch"'; exit 2; }
  log "\"event\":\"parent_ready\",\"seconds\":$(since $s)"
fi
rm -f "$SECRETS/rl-parent"

# The SGLang fork venv (x86_64 lock + editable fork source + Rust radix-tree core), cached on the volume: RL actors
# with `--engine sglang` and the sglang_bench stage shard. Sets SGL_VENV/SGL_COMMIT.
install_sglang() {
  local NEEDS_SGLANG=1 s
  if [ "$NEEDS_SGLANG" = 1 ]; then
    SGL_DIR=/home/werg/sglang-natlang
    SGL_VENV=$(python3 -c "import json;print(json.load(open('$CODE/training/neuralese/sglang-fork.json'))['python'])" | xargs dirname | xargs dirname)
    SGL_COMMIT=$(python3 -c "import json;print(json.load(open('$CODE/training/neuralese/sglang-fork.json'))['commit'])")
  fi
  if [ "$NEEDS_SGLANG" = 1 ] && [ ! -f "$SGL_VENV/.natlang-$SGL_COMMIT" ]; then
    s=$(date +%s)
    rm -rf "$SGL_DIR" "$SGL_VENV"; mkdir -p "$SGL_DIR"
    tar -xzf "$INPUTS/sglang-natlang-$SGL_COMMIT.tar.gz" -C "$SGL_DIR" || { log '"event":"failed","step":"fork-source"'; exit 2; }
    python3 -m pip install -q --no-cache-dir uv >/dev/null 2>&1 || { log '"event":"failed","step":"uv"'; exit 2; }
    export -n PIP_CONSTRAINT 2>/dev/null; unset PIP_CONSTRAINT
    python3 -m uv venv -q -p "$(command -v python3)" "$SGL_VENV" || { log '"event":"failed","step":"sglang-venv"'; exit 2; }
    python3 -m uv pip install -q -p "$SGL_VENV/bin/python" -r "$INPUTS/sglang-lock-$SGL_COMMIT-x86_64.txt" \
      --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match > "$OUT/sglang-install.log" 2>&1 \
      || { tail -20 "$OUT/sglang-install.log"; log '"event":"failed","step":"sglang-deps"'; exit 2; }
    # The fork's Rust radix-tree core (SGLang's default unified tree backend) is built here; the other Rust extensions
    # (gRPC, Rust server, multimodal) are unused by the actor. The loader may rebuild it for the installed torch, so
    # cargo stays on PATH for the actor too.
    if ! command -v cargo >/dev/null 2>&1 && [ ! -x "$HOME/.cargo/bin/cargo" ]; then
      (curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal -q) > "$OUT/rustup.log" 2>&1 \
        || { tail -20 "$OUT/rustup.log"; log '"event":"failed","step":"rustup"'; exit 2; }
    fi
    export PATH="$HOME/.cargo/bin:$PATH"
    SGLANG_BUILD_RUST_EXTS=mem_cache python3 -m uv pip install -q -p "$SGL_VENV/bin/python" --no-deps -e "$SGL_DIR/python" >> "$OUT/sglang-install.log" 2>&1 \
      || { tail -20 "$OUT/sglang-install.log"; log '"event":"failed","step":"sglang-fork"'; exit 2; }
    "$SGL_VENV/bin/python" -c "import sglang.srt.mem_cache.rust_tree_core.extension as e; e.load_tree_core_extension()" >> "$OUT/sglang-install.log" 2>&1 \
      || { tail -20 "$OUT/sglang-install.log"; log '"event":"failed","step":"sglang-rust-tree-core"'; exit 2; }
    KERNEL=sgl_kernel; nvidia-smi >/dev/null 2>&1 || KERNEL=sglang  # no driver in a CPU check sandbox
    "$SGL_VENV/bin/python" -c "import sglang, $KERNEL, torch; print(sglang.__file__, torch.__version__, torch.cuda.is_available())" >> "$OUT/sglang-install.log" 2>&1 \
      || { tail -20 "$OUT/sglang-install.log"; log '"event":"failed","step":"sglang-import"'; exit 2; }
    touch "$SGL_VENV/.natlang-$SGL_COMMIT"
    log "\"event\":\"sglang_ready\",\"commit\":\"$SGL_COMMIT\",\"seconds\":$(since $s)"
  fi
}

# A tune shard (tune.json): natlang_neuralese.tune.stage --shard I/N on the restored parent with the code snapshot
# (its ts-host natlang CLI and native student), the stage directory staged while it runs and once at the end; the
# DGX exporter shuts the sandbox down once the final export (status done or failed) is verified.
if [ -f "$L/tune.json" ]; then
  VENV=/home/werg/natlang/.venv-neuralese
  [ -x "$VENV/bin/python" ] || python3 -m venv --system-site-packages "$VENV" || { log '"event":"failed","step":"learner-venv"'; exit 2; }
  if [ ! -f "$CODE/node_modules/.natlang-npm-ci" ]; then
    (cd "$CODE" && npm ci --no-audit --no-fund > "$OUT/npm-root.log" 2>&1) || { tail -20 "$OUT/npm-root.log"; log '"event":"failed","step":"npm-root"'; exit 2; }
    touch "$CODE/node_modules/.natlang-npm-ci"
  fi
  [ -f "$CODE/ts-host/bin/natlang.mjs" ] && [ -d "$CODE/ts-host/dist" ] || { log '"event":"failed","step":"ts-host"'; exit 2; }
  export NATLANG_REPO=$CODE PYTHONPATH=$CODE/training/neuralese:$CODE PATH=$VENV/bin:$PATH
  export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 NATLANG_HASH_CACHE=$ROOT/hash-cache.jsonl NATLANG_NODE=$(command -v node)
  export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
  STAGE=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['stage_dir'])" "$L/tune.json")
  MODULE=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['module'])" "$L/tune.json")
  if [ "$MODULE" = natlang_neuralese.serve.sglang_bench ]; then install_sglang || exit 2; fi
  mapfile -d '' ARGS < <(python3 -c "import json,sys;sys.stdout.write(''.join(a+chr(0) for a in json.load(open(sys.argv[1]))['argv']))" "$L/tune.json")
  log "\"event\":\"tune_setup_done\",\"stage\":\"$STAGE\",\"node\":\"$(node --version 2>/dev/null)\",\"seconds\":$(since $t0)"
  await_upload() { local seq; seq=$(ls "$E" | grep -E '^[0-9]{6}\.json$' | tail -1 | cut -c1-6)
    log "\"event\":\"await_export\",\"seq\":\"$seq\""; while [ -n "$seq" ] && [ ! -f "$E/$seq.uploaded" ]; do sleep 30; done; }
  if [ "${NATLANG_RL_CHECK:-0}" = 1 ]; then  # CPU check: the stage's dry run in a scratch stage directory
    CHECK=$STAGE-check; mkdir -p "$CHECK"; cp -a "$STAGE/programs" "$CHECK/"; rm -f "$CHECK/programs/programs-prepared.json"
    for i in "${!ARGS[@]}"; do [ "${ARGS[$i]}" = --out ] && ARGS[$((i+1))]=$CHECK; done
    (cd "$CODE" && "$VENV/bin/python" -u -m "$MODULE" "${ARGS[@]}" --dry-run) > "$OUT/tune-check.log" 2>&1
    rc=$?; tail -20 "$OUT/tune-check.log"; log "\"event\":\"check_tune_dry_run\",\"rc\":$rc"
    python3 "$HELPER" stage "$L" "$E" >> "$OUT/exports.log" 2>&1; log "\"event\":\"check_staged\",\"rc\":$?"
    await_upload; log "\"event\":\"check_done\",\"seconds\":$(since $t0)"; exit 0
  fi
  python3 -c "import json,sys,time;json.dump({'status':'running','started':time.time()},open(sys.argv[1],'w'))" "$L/tune-status.json"
  ( while sleep "${NATLANG_CW_STAGE_SECONDS:-1800}"; do python3 "$HELPER" stage "$L" "$E" >> "$OUT/exports.log" 2>&1
      python3 "$HELPER" prune "$L" "$E" >> "$OUT/exports.log" 2>&1; done ) & stager=$!
  child=
  trap 'log "\"event\":\"term\""; kill $stager 2>/dev/null; [ -n "$child" ] && kill -TERM $child 2>/dev/null; wait $child 2>/dev/null; exit 143' TERM INT
  s=$(date +%s); log "\"event\":\"tune_start\",\"argv\":$(python3 -c "import json,sys;print(json.dumps(json.load(open(sys.argv[1]))['argv']))" "$L/tune.json")"
  (cd "$CODE" && exec "$VENV/bin/python" -u -m "$MODULE" "${ARGS[@]}") >> "$OUT/tune.log" 2>&1 &
  child=$!; wait $child; rc=$?; child=
  kill $stager 2>/dev/null; wait $stager 2>/dev/null
  status=done; [ $rc -eq 0 ] || status=failed
  python3 -c "import json,sys,time;v=json.load(open(sys.argv[1]));v.update(status=sys.argv[2],rc=int(sys.argv[3]),finished=time.time());json.dump(v,open(sys.argv[1],'w'))" "$L/tune-status.json" "$status" "$rc"
  log "\"event\":\"tune_done\",\"rc\":$rc,\"status\":\"$status\",\"seconds\":$(since $s),\"memory_peak\":\"$(cat /sys/fs/cgroup/memory.peak 2>/dev/null)\""
  [ $rc -eq 0 ] || tail -30 "$OUT/tune.log"
  python3 "$HELPER" stage "$L" "$E" >> "$OUT/exports.log" 2>&1 || log '"event":"stage_failed"'
  await_upload; log "\"event\":\"done\",\"status\":\"$status\",\"seconds\":$(since $t0)"
  [ $rc -eq 0 ] && exit 0 || exit 3
fi
PROGRESS=$(progress) || { log '"event":"failed","step":"progress"'; exit 2; }
CFG=$(field "$PROGRESS" config); KIND=$(field "$PROGRESS" kind)

# 2. environments
VENV=/home/werg/natlang/.venv-neuralese
if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv --system-site-packages "$VENV" || { log '"event":"failed","step":"learner-venv"'; exit 2; }
fi
# rl_cloud.actor_uses_sglang: an explicit --engine sglang actor, or a raw LFM native-python actor that
# production.adopt_actor_engine_defaults switches to SGLang at its next unbound collection.
NEEDS_SGLANG=$(python3 -c "
import json,sys;c=json.load(open(sys.argv[1]));a=c.get('actor_command') or []
e=a[a.index('--engine')+1] if '--engine' in a[:-1] else 'native'
adopt=bool(a) and e=='native' and not (c.get('actor_engine_keep_native') is True or c.get('coefficient_policy') or c.get('actor_engine','native-python')!='native-python' or c.get('model')!='lfm')
print(int(not c.get('mandatory_sft_only') and (e=='sglang' or adopt)))" "$CFG")
SGL_VENV=; SGL_COMMIT=
[ "$NEEDS_SGLANG" = 1 ] && install_sglang
log "\"event\":\"actor_environment\",\"sglang\":$NEEDS_SGLANG,\"kind\":\"$KIND\""

# 3. runtime: the config's frozen runtime as restored, or ROOT/runtime from the bundle's code snapshot
R=$(python3 -c "
import json,sys,pathlib;c=json.load(open(sys.argv[1]))
print((c.get('actor_environment') or {}).get('NATLANG_REPO') or pathlib.Path(c['runtime']).parent)" "$CFG")
BUILT_RUNTIME=0
if [ "$R" = "$ROOT/runtime" ]; then
  BUILT_RUNTIME=1
  COMMIT=$(cat "$CODE/.coreweave-code-commit" 2>/dev/null || python3 -c "import json;print(json.load(open('${NATLANG_CW_CONTROL_DIR}/bundle.json'))['code']['git_sha'])")
  if [ "$(cat "$R/COMMIT" 2>/dev/null)" != "$COMMIT" ]; then
    s=$(date +%s)
    (cd "$CODE" && npm ci --no-audit --no-fund > "$OUT/npm-root.log" 2>&1) || { tail -20 "$OUT/npm-root.log"; log '"event":"failed","step":"npm-root"'; exit 2; }
    if [ ! -f "$CODE/ts-host/dist/native/prompt.js" ]; then
      (cd "$CODE/ts-host" && npm ci --no-audit --no-fund && npm run -s build:node) > "$OUT/tshost.log" 2>&1 \
        || { tail -20 "$OUT/tshost.log"; log '"event":"failed","step":"ts-host"'; exit 2; }
    fi
    rm -rf "$R.partial" "$R"; mkdir -p "$R.partial"
    (cd "$CODE" && tar --exclude=./.venv-neuralese --exclude=./.coreweave-prebuilt -cf - .) | tar -x -C "$R.partial"
    cp -an "$INPUTS/runtime-runs/." "$R.partial/runs/"
    echo "$COMMIT" > "$R.partial/COMMIT"; mv "$R.partial" "$R"
    log "\"event\":\"runtime_ready\",\"commit\":\"$COMMIT\",\"seconds\":$(since $s)"
  fi
elif [ ! -d "$R/training/neuralese/natlang_neuralese" ]; then
  log "\"event\":\"failed\",\"step\":\"frozen-runtime\",\"runtime\":\"$R\""; exit 2
else
  log "\"event\":\"frozen_runtime\",\"runtime\":\"$R\",\"commit\":\"$(python3 -c "import json;print(json.load(open('$R/runtime-input-closure.json')).get('commit',''))" 2>/dev/null)\""
fi
NODE_BIN=$(dirname "$(command -v node 2>/dev/null || echo /usr/bin/node)")
[ -n "$SGL_VENV" ] && export PATH="$HOME/.cargo/bin:$PATH"
log "\"event\":\"setup_done\",\"seconds\":$(since $t0)"

# 4. controller calls
export NATLANG_REPO=$R PYTHONPATH=$R/training/neuralese:$R PATH=$NODE_BIN:$VENV/bin:$PATH
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 NATLANG_HASH_CACHE=$ROOT/hash-cache.jsonl
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
# Sandbox-owned checkpoint controls (the DGX's host admission wrapper and update lock are not part of a cloud config):
# the recovery free-space floor on the volume, and a discrete GPU (CUDA memory is not host memory).
export NATLANG_CHECKPOINT_MIN_FREE_BYTES=${NATLANG_CHECKPOINT_MIN_FREE_BYTES:-42949672960} NATLANG_DISCRETE_CUDA_MEMORY=1

# NATLANG_RL_CHECK=1 (a CPU sandbox, rl_cloud.py launch --check): everything but the GPU work -- environments,
# runtime, parent and inputs, custody load, imports, one export of the boundary through the DGX exporter -- then exit.
if [ "${NATLANG_RL_CHECK:-0}" = 1 ]; then
  (cd "$R" && "$VENV/bin/python" -c 'import sys; from natlang_neuralese.common.custody import ProtectedWorlds; ProtectedWorlds.load(root=sys.argv[1]); import natlang_neuralese.rl.production, natlang_neuralese.rl.native_stage, natlang_neuralese.train.native_state_sft' "$R") \
    > "$OUT/check-learner.log" 2>&1; log "\"event\":\"check_learner_imports\",\"rc\":$?"
  if [ "$NEEDS_SGLANG" = 1 ]; then
    "$SGL_VENV/bin/python" -c 'import sglang, sglang.srt.models.lfm3, sglang.srt.layers.sampler; print(sglang.__file__)' \
      > "$OUT/check-sglang.log" 2>&1; log "\"event\":\"check_sglang_imports\",\"rc\":$?"
  fi
  (cd "$R" && "$VENV/bin/python" -c 'import sys,json; from natlang_neuralese.train.recipe import load_recipe; c=json.load(open(sys.argv[1])); load_recipe(c["production_recipe"]); print("recipe ok")' "$CFG") \
    > "$OUT/check-recipe.log" 2>&1; log "\"event\":\"check_recipe\",\"rc\":$?"
  python3 "$HELPER" stage "$L" "$E" >> "$OUT/exports.log" 2>&1; log "\"event\":\"check_staged\",\"rc\":$?"
  seq=$(ls "$E" | grep -E '^[0-9]{6}\.json$' | tail -1 | cut -c1-6)
  while [ -n "$seq" ] && [ ! -f "$E/$seq.uploaded" ]; do sleep 20; done
  log "\"event\":\"check_done\",\"seconds\":$(since $t0)"; exit 0
fi

rebind() {  # apply_pending_switch.sh semantics: an unbound round takes this sandbox's built runtime (actor entry, inventory)
  [ "$BUILT_RUNTIME" = 1 ] || return 0
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
# The next SFT window from the active window's earned boundary (the lineage runtime's own continuation_config).
advance_window() {
  local p n out id
  p=$(progress); n=$(python3 -c "import json;print(len(json.load(open('$L/lineage.json'))['windows']))")
  out=$L/w$(printf %06d "$n"); id="$(python3 -c "import json;print(json.load(open('$CFG')).get('id','lfm3'))" | sed 's/-w[0-9]\{6\}$//')-w$(printf %06d "$n")"
  (cd "$R" && "$VENV/bin/python" "$WINDOW" next --previous-config "$CFG" --out "$out" --config-out "$out/config.json" --id "$id") \
    >> "$OUT/windows.log" 2>&1 || { tail -20 "$OUT/windows.log"; return 1; }
  python3 "$HELPER" activate "$L" "$out" "$out/config.json" >> "$OUT/windows.log" 2>&1 || return 1
  CFG=$out/config.json
  log "\"event\":\"window_prepared\",\"window\":$n,\"config\":\"$CFG\""
}
# An interrupted window resumes in its own earned slot (natlang.rolling-sft-resume/1), never from its old parent.
bind_rolling() {
  local active new
  active=$(field "$(progress)" out); new=$active/config-rolling-$(date -u +%Y%m%dT%H%M%SZ).json
  (cd "$R" && "$VENV/bin/python" "$WINDOW" rolling --config "$CFG" --config-out "$new") >> "$OUT/windows.log" 2>&1 \
    || { tail -20 "$OUT/windows.log"; return 1; }
  python3 "$HELPER" activate "$L" "$active" "$new" >> "$OUT/windows.log" 2>&1 || return 1
  CFG=$new
  log "\"event\":\"rolling_resume_bound\",\"config\":\"$CFG\""
}
stager=
stage_periodically() {  # an SFT window trains for hours: its rolling checkpoints are exported while it runs
  while sleep "${NATLANG_CW_STAGE_SECONDS:-1800}"; do
    python3 "$HELPER" stage "$L" "$E" >> "$OUT/exports.log" 2>&1 || log '"event":"stage_failed"'
    python3 "$HELPER" prune "$L" "$E" >> "$OUT/exports.log" 2>&1 || true
  done
}
stop_stager() { [ -n "$stager" ] && { kill "$stager" 2>/dev/null; wait "$stager" 2>/dev/null; stager=; }; }
child=
trap 'log "\"event\":\"term\""; stop_stager; [ -n "$child" ] && kill -TERM $child 2>/dev/null; wait $child 2>/dev/null; exit 143' TERM INT
i=0
while [ "$ROUNDS" -eq 0 ] || [ $i -lt "$ROUNDS" ]; do
  P=$(progress); CFG=$(field "$P" config); A=$(field "$P" out)
  if [ "$KIND" = sft ]; then
    if [ "$(field "$P" window_complete)" = true ]; then  # resumed at a completed window's boundary
      advance_window || { log '"event":"failed","step":"next window"'; exit 3; }; continue
    fi
    if [ "$(field "$P" rolling_slot)" = true ] && [ "$(field "$P" bound_rolling_resume)" != true ]; then
      bind_rolling || { log '"event":"failed","step":"rolling resume"'; exit 3; }
    fi
    export NATLANG_NATIVE_PROFILE_FIRST_UPDATE=$A/learner/first-update-profile.json
    export NATLANG_POLICY_LOGPROB_RECEIPT=$A/learner/behavior-replay-logprobs.json
    stage_periodically & stager=$!
  else
    rebind || { log '"event":"failed","step":"actor rebind"'; exit 3; }
  fi
  before=$(field "$P" trained_rounds); s=$(date +%s)
  log "\"event\":\"call_start\",\"kind\":\"$KIND\",\"controller\":\"$A\",\"trained_rounds\":$before"
  (cd "$R" && exec "$VENV/bin/python" -u -m natlang_neuralese.rl.production run --config "$CFG" --rounds 1) \
    >> "$OUT/controller.log" 2>&1 &
  child=$!; wait $child; rc=$?; child=
  stop_stager
  P=$(progress)
  if [ "$KIND" = sft ]; then advanced=$(field "$P" window_complete)
  else advanced=$( [ "$(field "$P" trained_rounds)" -gt "$before" ] && echo true || echo false ); fi
  log "\"event\":\"call_done\",\"rc\":$rc,\"phase\":\"$(field "$P" phase)\",\"advanced\":$advanced,\"trained_rounds\":$(field "$P" trained_rounds),\"window_updates\":\"$(field "$P" window_updates)\",\"seconds\":$(since $s),\"memory_peak\":\"$(cat /sys/fs/cgroup/memory.peak 2>/dev/null)\""
  # Whatever the outcome, the boundary (and an interrupted window's rolling slot) is exported.
  python3 "$HELPER" stage "$L" "$E" >> "$OUT/exports.log" 2>&1 || log '"event":"stage_failed"'
  if [ $rc -ne 0 ] || [ "$advanced" != true ]; then
    tail -30 "$OUT/controller.log"
    log "\"event\":\"EXIT\",\"reason\":\"did not advance\",\"rc\":$rc"; exit 3
  fi
  python3 "$HELPER" prune "$L" "$E" >> "$OUT/exports.log" 2>&1 || true
  i=$((i+1))
  if [ "$KIND" = sft ] && { [ "$ROUNDS" -eq 0 ] || [ $i -lt "$ROUNDS" ]; }; then
    advance_window || { log '"event":"failed","step":"next window"'; exit 3; }
  fi
done
# the last staged export must reach the DGX before the job ends
seq=$(ls "$E" | grep -E '^[0-9]{6}\.json$' | tail -1 | cut -c1-6)
log "\"event\":\"await_export\",\"seq\":\"$seq\""
while [ -n "$seq" ] && [ ! -f "$E/$seq.uploaded" ]; do sleep 30; done
log "\"event\":\"done\",\"seconds\":$(since $t0)"
