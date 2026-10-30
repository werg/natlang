#!/bin/bash
# LFM3 decision eval on one CoreWeave GPU sandbox (plans/neuralese/LFM3_DECISION_EVAL_20261029.md): the 128-case
# deterministic gate (scripts/neuralese_data/lfm3_checkpoint_gate.py --serving deterministic --repeat-base) of several
# trained GGUF exports against the official LFM3-1.7B BF16 base. Job: scripts/coreweave/jobs/lfm3/decision-eval.job.json.
#
#   lfm3_decision_eval.sh OUT LLAMA_SOURCE_TAR LLAMA_GIT_SHA MODEL_DIR [MODEL_DIR ...]
#
# 1. llama-server of the pinned fork, built here for the assigned GPU's compute capability only (the flags of
#    scripts/coreweave/prebuilt_build.sh); its sha256 goes into OUT/build.json.
# 2. The base GGUF from the hub at the pinned revision, sha256 checked.
# 3. Each MODEL_DIR is a GGUF export directory (manifest.json, chat template, hf-config) whose model-bf16.gguf was
#    shipped as an XOR delta against a basis built from the base (scripts/coreweave/gguf_basis.py,
#    lossless_file_transfer.py): the basis and the restored GGUF are both verified against their recorded sha256.
# 4. CUDA MPS when available (kernels and numerics are unchanged; it lets the single-sequence servers' kernels overlap),
#    then the gate with as many single-sequence servers per arm as the free GPU memory allows (LFM3_GATE_SERVERS
#    overrides; ~6 GB per server).
# Results: OUT/gate (decision.json, per-model eval-deterministic/), OUT/base-runs (the two base executions).
set -uo pipefail
OUT=$1; LLAMA_TAR=$2; LLAMA_SHA=$3; shift 3
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
exec > >(tee -a "$OUT/driver.log") 2>&1
WORK=${NATLANG_CW_WORK:-/work}
BASE_URL=https://huggingface.co/LiquidAI/LFM3-1.7B-GGUF/resolve/b3f2ae5d73cd3c91bf40e8ba752c648aa05dffdf/LFM3-1.7B-BF16.gguf
BASE_SHA=39158a79f56edb85e0c4252ce6447b8d7883ef82ffedd02f0cda7c84da5e9f68
log() { echo "{\"at\":\"$(date -u +%FT%TZ)\",$1}"; }
facts() {
  { echo "== at $(date -u +%FT%TZ)"; nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu,driver_version,compute_cap --format=csv
    echo "nproc $(nproc)"; echo "cgroup memory.max $(cat /sys/fs/cgroup/memory.max 2>/dev/null)"; echo "cgroup cpu.max $(cat /sys/fs/cgroup/cpu.max 2>/dev/null)"
    free -g; lscpu | grep -E 'Model name|^CPU\(s\)'; df -h "$WORK"; } >> "$OUT/host.txt" 2>&1
}
facts
cd "$NATLANG_REPO" || exit 2

# 1. llama-server for this GPU
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' .')
ARCH="${CC}-real"; [ "$CC" -ge 90 ] && ARCH="${CC}a-real"  # sm_90a/100a/120a as in the registered prebuilt builds
SRC=$WORK/llama-src; BUILD=$WORK/llama-build
if [ ! -x "$BUILD/bin/llama-server" ]; then
  rm -rf "$SRC" "$BUILD"; mkdir -p "$SRC"; tar -xzf "$LLAMA_TAR" -C "$SRC" || exit 3
  t0=$(date +%s)
  cmake -S "$SRC/llama.cpp" -B "$BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
    "-DCMAKE_CUDA_ARCHITECTURES=$ARCH" -DGGML_NATIVE=OFF -DGGML_CUDA_NCCL=OFF -DBUILD_SHARED_LIBS=OFF \
    -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF > "$OUT/cmake.log" 2>&1 || { tail -30 "$OUT/cmake.log"; exit 3; }
  cmake --build "$BUILD" -j"$(nproc)" --target llama-server > "$OUT/ninja.log" 2>&1 || { tail -30 "$OUT/ninja.log"; exit 3; }
  log "\"event\":\"built\",\"arch\":\"$ARCH\",\"seconds\":$(( $(date +%s) - t0 ))"
fi
SERVER=$BUILD/bin/llama-server
python3 - "$OUT/build.json" "$LLAMA_SHA" "$ARCH" "$SERVER" "$LLAMA_TAR" <<'PY'
import hashlib, json, sys
path, sha, arch, server, tar = sys.argv[1:]
digest = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
json.dump({'llama_cpp_git_sha': sha, 'source_tar_sha256': digest(tar), 'cuda_architectures': arch,
           'llama_server_sha256': digest(server),
           'cmake': ['-DGGML_CUDA=ON', f'-DCMAKE_CUDA_ARCHITECTURES={arch}', '-DGGML_NATIVE=OFF', '-DGGML_CUDA_NCCL=OFF',
                     '-DBUILD_SHARED_LIBS=OFF', '-DLLAMA_OPENSSL=OFF', '-DCMAKE_BUILD_TYPE=Release']},
          open(path, 'w'), indent=1)
PY
"$SERVER" --version >> "$OUT/build.json.version" 2>&1

# 2. base GGUF
BASE=$WORK/models/LFM3-1.7B-BF16.gguf
if [ ! -f "$BASE" ]; then
  mkdir -p "$(dirname "$BASE")"; curl -fsSL --retry 4 -o "$BASE.partial" "$BASE_URL" && mv "$BASE.partial" "$BASE" || exit 4
fi
echo "$BASE_SHA  $BASE" | sha256sum -c - || exit 4

# 3. trained GGUFs from their deltas
for dir in "$@"; do
  [ -f "$dir/model-bf16.gguf" ] && continue
  python3 - "$dir" "$BASE" <<'PY' || exit 5
import json, os, subprocess, sys
sys.path.insert(0, os.environ['NATLANG_REPO'])
from scripts.coreweave import lossless_file_transfer as codec
directory, base = sys.argv[1:]
delta = json.load(open(f'{directory}/delta.json'))
basis = f'{directory}/basis.tmp'
subprocess.run([sys.executable, f"{os.environ['NATLANG_REPO']}/scripts/coreweave/gguf_basis.py", 'build',
                f'{directory}/basis-layout.json', base, basis], check=True)
manifest = json.load(open(f'{directory}/manifest.json'))
target = {'sha256': manifest['files'][0]['sha256'], 'bytes': manifest['files'][0]['bytes']}
assert target == delta['target'], 'delta does not bind the manifest GGUF'
codec.decode(f'{directory}/model-bf16.gguf.xor.zst', basis, f'{directory}/model-bf16.gguf',
             container_identity=delta['container'], target_identity=target, basis_identity=delta['basis'])
os.unlink(basis)
print(json.dumps({'restored': directory, **target}))
PY
done

# 4. MPS and servers per arm
export CUDA_MPS_PIPE_DIRECTORY=$WORK/mps-pipe CUDA_MPS_LOG_DIRECTORY=$WORK/mps-log
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
MPS=0
if command -v nvidia-cuda-mps-control >/dev/null && nvidia-cuda-mps-control -d; then
  if timeout 120 "$SERVER" --version >/dev/null 2>&1 && python3 -c "import torch; torch.ones(1, device='cuda').sum().item()" 2>/dev/null; then
    MPS=1
  else
    echo quit | nvidia-cuda-mps-control; unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY
  fi
else
  unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY
fi
ARMS=$(( $# + 2 ))
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
SERVERS=${LFM3_GATE_SERVERS:-$(( (FREE / 1024 - 4) / (6 * ARMS) ))}
[ "$SERVERS" -lt 1 ] && SERVERS=1; [ "$SERVERS" -gt 4 ] && SERVERS=4
log "\"event\":\"gate\",\"mps\":$MPS,\"free_mib\":$FREE,\"arms\":$ARMS,\"servers_per_arm\":$SERVERS"
( while sleep 600; do facts; done ) &
FACTS=$!
NATLANG_HASH_CACHE=$WORK/hash-cache.json python3 scripts/neuralese_data/lfm3_checkpoint_gate.py "$@" --out "$OUT/gate" \
  --serving deterministic --repeat-base --servers "$SERVERS" --llama-server "$SERVER" --base-gguf "$BASE" \
  --eval-root "$WORK/gate-root" --port 8170 > "$OUT/gate.log" 2>&1
rc=$?
kill $FACTS 2>/dev/null
facts
mkdir -p "$OUT/base-runs"
for run in "$WORK"/gate-root/base-*; do [ -d "$run" ] && cp -r "$run" "$OUT/base-runs/"; done
[ "$MPS" = 1 ] && echo quit | nvidia-cuda-mps-control
log "\"event\":\"done\",\"exit\":$rc"
tail -3 "$OUT/gate.log"
exit $rc
