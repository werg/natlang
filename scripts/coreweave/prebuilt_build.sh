#!/bin/bash
# Build the x86_64 prebuilt artifacts for CoreWeave sandboxes inside the exact runtime image (a CPU-only sandbox; see
# scripts/coreweave/jobs/prebuilt-build.json and scripts/coreweave/prebuilt.py). DGX is aarch64, so x86_64 CUDA
# binaries and wheels cannot be built there.
#
#   prebuilt_build.sh OUT LLAMA_SOURCE_TAR LLAMA_GIT_SHA [FAT_ARCHS]
#
# Produces in OUT (deterministic tar.gz: sorted names, mtime 0, numeric owner 0, gzip -n):
#   llama-cpp-neuralese-<sha12>-sm120.tar.gz  llama-server + neuralese tools, CUDA 120a-real, static ggml/llama libs
#   llama-cpp-neuralese-<sha12>-fat.tar.gz    same for FAT_ARCHS (default 90a-real;100a-real;120a-real), if it builds
#   wheelhouse-<image12>.tar.gz               exact wheels for training/neuralese/docker/requirements.txt as resolved
#                                             against the image's own pip constraints (install --no-index --no-deps)
#   build.json                                timings, host facts, artifact sha256/bytes and the exact build flags
# Consumers must run the same image digest: the CUDA runtime/cuBLAS libraries come from the image (ldd recorded).
set -euo pipefail
OUT=$1; LLAMA_TAR=$2; LLAMA_SHA=$3; FAT_ARCHS=${4:-"90a-real;100a-real;120a-real"}
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
exec > >(tee -a "$OUT/build.log") 2>&1
B=${NATLANG_CW_WORK:-/work}/prebuilt-build; rm -rf "$B"; mkdir -p "$B"
now() { date +%s.%N; }
facts="$OUT/host.txt"
{ uname -a; nproc; free -g; lscpu | sed -n '1,20p'; df -h "$B"; command -v nvcc && nvcc --version; cmake --version | head -1
  command -v ninja && ninja --version; command -v zstd || echo "zstd: absent"; python3 --version
  echo "PIP_CONSTRAINT=${PIP_CONSTRAINT:-}"; ls /etc/pip/constraint.txt 2>/dev/null; } > "$facts" 2>&1 || true
cat "$facts"
detar() { tar --sort=name --mtime=@0 --owner=0 --group=0 --numeric-owner -C "$1" -cf - "${@:2}" | gzip -n -9 > "$OUT/$ARTIFACT"; }
SHA12=${LLAMA_SHA:0:12}

# 1. llama.cpp fork: one configure/build per arch set; the ggml/llama libraries are static, so bin/ is self-contained
#    apart from the image's CUDA runtime and the driver.
build_llama() {  # $1 label, $2 archs
  local label=$1 archs=$2 src="$B/src-$1" build="$B/build-$1" started
  mkdir -p "$src"; tar -xzf "$LLAMA_TAR" -C "$src"
  started=$(now)
  cmake -S "$src/llama.cpp" -B "$build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
    "-DCMAKE_CUDA_ARCHITECTURES=$archs" -DGGML_NATIVE=OFF -DGGML_CUDA_NCCL=OFF -DBUILD_SHARED_LIBS=OFF \
    -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF > "$OUT/cmake-$label.log" 2>&1
  cmake --build "$build" -j"$(nproc)" --target llama-server llama-state-span-dump llama-logits-dump llama-bench \
    llama-quantize llama-cli llama-perplexity > "$OUT/ninja-$label.log" 2>&1
  local seconds; seconds=$(python3 -c "print(round($(now) - $started, 1))")
  local stage="$B/stage-$label/llama-cpp-neuralese"; mkdir -p "$stage/bin"
  for t in llama-server llama-state-span-dump llama-logits-dump llama-bench llama-quantize llama-cli llama-perplexity; do
    cp "$build/bin/$t" "$stage/bin/$t"; strip --strip-unneeded "$stage/bin/$t" || true
  done
  ldd "$stage/bin/llama-server" > "$stage/ldd-llama-server.txt" 2>&1 || true
  python3 - "$stage/BUILD.json" "$LLAMA_SHA" "$archs" "$seconds" <<'PY'
import json, sys
path, sha, archs, seconds = sys.argv[1:]
json.dump({'source': 'github.com/werg/llama.cpp-neuralese', 'branch': 'neuralese-lfm3', 'git_sha': sha,
           'cuda_architectures': archs, 'build_seconds': float(seconds),
           'cmake': ['-DGGML_CUDA=ON', f'-DCMAKE_CUDA_ARCHITECTURES={archs}', '-DGGML_NATIVE=OFF', '-DGGML_CUDA_NCCL=OFF',
                     '-DBUILD_SHARED_LIBS=OFF', '-DLLAMA_OPENSSL=OFF', '-DCMAKE_BUILD_TYPE=Release']},
          open(path, 'w'), indent=1)
PY
  ARTIFACT="llama-cpp-neuralese-$SHA12-$label.tar.gz"; detar "$B/stage-$label" llama-cpp-neuralese
  echo "{\"artifact\":\"$ARTIFACT\",\"build_seconds\":$seconds}" >> "$OUT/timings.jsonl"
}
build_llama sm120 "120a-real"

# 2. wheelhouse: what the bootstrap's pip install adds to this image, pinned to the versions it resolved.
started=$(now)
python3 -m pip freeze --all > "$OUT/freeze-base.txt"
python3 -m pip install -q --no-cache-dir -r "$NATLANG_REPO/training/neuralese/docker/requirements.txt"
python3 -m pip freeze --all > "$OUT/freeze-after.txt"
python3 - "$OUT/freeze-base.txt" "$OUT/freeze-after.txt" "$B/wheelhouse/added.txt" <<'PY'
import os, sys
base, after = (set(open(p).read().split()) for p in sys.argv[1:3])
added = sorted(line for line in after - base if '==' in line)
os.makedirs(os.path.dirname(sys.argv[3]), exist_ok=True)
open(sys.argv[3], 'w').write(''.join(f'{line}\n' for line in added))
print('added', len(added), 'packages')
PY
(cd "$B/wheelhouse" && PIP_CONSTRAINT= python3 -m pip download -q --no-deps --only-binary=:all: -d . -r added.txt) \
  || (cd "$B/wheelhouse" && PIP_CONSTRAINT= python3 -m pip wheel -q --no-deps -w . -r added.txt)
cp "$NATLANG_REPO/training/neuralese/docker/requirements.txt" "$B/wheelhouse/requirements.txt"
cp "$OUT/freeze-after.txt" "$B/wheelhouse/freeze-after.txt"
IMAGE12=${NATLANG_CW_IMAGE_DIGEST:-unknown}; IMAGE12=${IMAGE12#sha256:}; IMAGE12=${IMAGE12:0:12}
ARTIFACT="wheelhouse-$IMAGE12.tar.gz"; detar "$B" wheelhouse
echo "{\"artifact\":\"$ARTIFACT\",\"build_seconds\":$(python3 -c "print(round($(now) - $started, 1))")}" >> "$OUT/timings.jsonl"

# 3. optional fat build (sm_90/100/120), bounded so it cannot hold the sandbox for long
if [ "${PREBUILT_FAT:-1}" = 1 ]; then
  timeout "${PREBUILT_FAT_SECONDS:-1800}" bash -c "set -euo pipefail; $(declare -f now build_llama detar); OUT='$OUT' B='$B' LLAMA_TAR='$LLAMA_TAR' \
    LLAMA_SHA='$LLAMA_SHA' SHA12='$SHA12' build_llama fat '$FAT_ARCHS'" || echo '{"artifact":"fat","failed_or_timed_out":true}' >> "$OUT/timings.jsonl"
fi

python3 - "$OUT" <<'PY'
import hashlib, json, os, sys
out = sys.argv[1]
arts = {}
for name in sorted(os.listdir(out)):
    if name.endswith('.tar.gz'):
        data = open(os.path.join(out, name), 'rb').read()
        arts[name] = {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
timings = [json.loads(l) for l in open(os.path.join(out, 'timings.jsonl'))]
json.dump({'artifacts': arts, 'timings': timings, 'image_digest': os.environ.get('NATLANG_CW_IMAGE_DIGEST'),
           'nproc': os.cpu_count()}, open(os.path.join(out, 'build.json'), 'w'), indent=1)
print(json.dumps(arts, indent=1))
PY
