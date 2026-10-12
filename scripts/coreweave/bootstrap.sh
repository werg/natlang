#!/bin/bash
# In-sandbox bootstrap for natlang CoreWeave jobs (mounted read-only at /opt/natlang/bootstrap.sh by cwctl.py launch;
# the sandbox's main process is `bash bootstrap.sh <job command...>`, so its stdout/stderr are the sandbox logs).
#
# Steps (each logged as one JSON line, so `cwsandbox logs` shows where a job is):
#   1. python deps: boto3 >= 1.42.5 (container-credential exchange against api.coreweave.com) and the training
#      requirements shared with the Docker image (training/neuralese/docker/requirements.txt, read from the code
#      snapshot, so image and bootstrap install one list). Skipped when NATLANG_CW_IMAGE_READY=1 (prebuilt image).
#   2. mirror roots: the DGX absolute roots (NATLANG_CW_MIRROR_ROOTS, default "/home/werg /mnt/external") become
#      symlinks into the scratch volume, so recipe plans, staged inputs and checkpoints that record absolute paths
#      resolve unchanged and NATLANG_REPO is the same path as on DGX.
#   3. code: the bundle manifest's code snapshot (git archive of one commit) is downloaded, its sha256 verified, and
#      unpacked at $NATLANG_REPO. No GitHub credential enters the sandbox.
#   4. inputs: scripts/coreweave/bundle.py fetch materializes and verifies every bundle file.
#   5. optional: NATLANG_CW_BUILD_TSHOST=1 builds ts-host (task-success evals run node ts-host/dist/...).
#   6. run the job command in $NATLANG_REPO/$NATLANG_CW_WORKDIR. On exit (or SIGTERM from `stop`, which gives the
#      job graceful_shutdown_seconds), every directory in NATLANG_CW_OUTPUTS is published to
#      $NATLANG_CW_BUCKET/$NATLANG_CW_OUTPUT_PREFIX<dir name>/ (content-addressed, verified on fetch).
#
# Environment (set by cwctl.py from the job; no credentials: object storage uses the sandbox's own WIF token):
#   NATLANG_CW_JOB NATLANG_CW_WORK NATLANG_CW_BUCKET NATLANG_CW_BUNDLE NATLANG_CW_OUTPUT_PREFIX NATLANG_CW_OUTPUTS
#   NATLANG_CW_WORKDIR NATLANG_CW_MIRROR_ROOTS NATLANG_CW_IMAGE_READY NATLANG_CW_BUILD_TSHOST NATLANG_CW_RESUME
#   NATLANG_CW_PUBLISH_INTERVAL NATLANG_CW_S3_ADDRESSING (path only for the local rehearsal) AWS_REGION
set -uo pipefail
WORK=${NATLANG_CW_WORK:-/work}
export NATLANG_REPO=${NATLANG_REPO:-/home/werg/natlang}
log() { printf '{"bootstrap":"%s","job":"%s","at":"%s"%s}\n' "$1" "${NATLANG_CW_JOB:-}" "$(date -u +%FT%TZ)" "${2:+,$2}"; }
die() { log failed "\"step\":\"$1\""; exit 0; }  # exit 0: see the end of this file

log start "\"sandbox\":\"${CW_SANDBOX_ID:-}\""
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null | sed 's/^/gpu: /'

# 1. client deps for object storage (the code snapshot's requirements come after step 3)
mkdir -p "$WORK"
if ! python3 -c "import boto3, sys; sys.exit(tuple(map(int, boto3.__version__.split('.')[:2])) < (1, 42))" 2>/dev/null; then
  python3 -m pip install -q --no-cache-dir 'boto3>=1.42.5' || die pip-boto3
fi

# 2. mirror the DGX absolute roots onto scratch
for root in ${NATLANG_CW_MIRROR_ROOTS:-/home/werg /mnt/external}; do
  if [ ! -e "$root" ]; then
    mkdir -p "$WORK/mirror$root" "$(dirname "$root")" && ln -s "$WORK/mirror$root" "$root" || die "mirror $root"
  fi
done
mkdir -p "$NATLANG_REPO"

# 3. code snapshot from the bundle manifest
if [ -n "${NATLANG_CW_BUNDLE:-}" ]; then
  python3 - <<'PY' || die code
import hashlib, json, os, tarfile, boto3
from botocore.client import Config
s3 = boto3.client('s3', endpoint_url=os.environ.get('AWS_ENDPOINT_URL_S3', 'https://cwobject.com'),
                  config=Config(region_name=os.environ['AWS_REGION'],
                                s3={'addressing_style': os.environ.get('NATLANG_CW_S3_ADDRESSING', 'virtual')}))
bucket, work, repo = os.environ['NATLANG_CW_BUCKET'], os.environ.get('NATLANG_CW_WORK', '/work'), os.environ['NATLANG_REPO']
manifest = json.loads(s3.get_object(Bucket=bucket, Key=os.environ['NATLANG_CW_BUNDLE'])['Body'].read())
os.makedirs(f'{work}/.coreweave', exist_ok=True)
with open(f'{work}/.coreweave/bundle.json', 'w') as f:
    json.dump(manifest, f)
code = manifest.get('code')
if code:
    path = f'{work}/.coreweave/code.tar.gz'
    s3.download_file(bucket, code['key'], path)
    digest = hashlib.sha256(open(path, 'rb').read()).hexdigest()
    if digest != code['sha256']:
        raise SystemExit(f'code snapshot sha256 {digest} != {code["sha256"]}')
    with tarfile.open(path) as tar:
        tar.extractall(repo, filter='data')
    print(json.dumps({'bootstrap': 'code', 'git_sha': code['git_sha'], 'sha256': digest}))
PY
  if [ "${NATLANG_CW_IMAGE_READY:-0}" != 1 ] && [ -f "$NATLANG_REPO/training/neuralese/docker/requirements.txt" ]; then
    python3 -m pip install -q --no-cache-dir -r "$NATLANG_REPO/training/neuralese/docker/requirements.txt" || die pip-requirements
  fi
  # 4. inputs, verified
  python3 "$NATLANG_REPO/scripts/coreweave/bundle.py" fetch --bucket "$NATLANG_CW_BUCKET" \
    --manifest "$WORK/.coreweave/bundle.json" --root "$NATLANG_REPO" || die fetch
fi

# 5. ts-host for task-success evals
if [ "${NATLANG_CW_BUILD_TSHOST:-0}" = 1 ]; then
  if ! node -e 'process.exit(+process.versions.node.split(".")[0] >= 22 ? 0 : 1)' 2>/dev/null; then
    # Node >= 22 from nodejs.org, checked against the release's SHASUMS256.txt (NATLANG_CW_NODE_VERSION pins it).
    NODE_VERSION=${NATLANG_CW_NODE_VERSION:-v24.0.1}; ARCH=$(uname -m | sed 's/x86_64/x64/;s/aarch64/arm64/')
    NODE_TAR=node-$NODE_VERSION-linux-$ARCH.tar.xz; NODE_URL=https://nodejs.org/dist/$NODE_VERSION
    (cd "$WORK" && curl -fsSLO "$NODE_URL/$NODE_TAR" && curl -fsSL "$NODE_URL/SHASUMS256.txt" | grep " $NODE_TAR\$" \
      | sha256sum -c - >/dev/null && tar -xJf "$NODE_TAR") || die node
    export PATH="$WORK/node-$NODE_VERSION-linux-$ARCH/bin:$PATH"; export NATLANG_NODE_DIR="$WORK/node-$NODE_VERSION-linux-$ARCH/bin"
    log node "\"version\":\"$NODE_VERSION\""
  fi
  (cd "$NATLANG_REPO/ts-host" && npm ci --no-audit --no-fund && npm run build) >/dev/null || die ts-host
  log ts-host
fi

BUNDLE="$NATLANG_REPO/scripts/coreweave/bundle.py"
outputs=()
for dir in ${NATLANG_CW_OUTPUTS:-}; do
  case "$dir" in /*) outputs+=("$dir") ;; *) outputs+=("$NATLANG_REPO/$dir") ;; esac
done

# 5b. resume: fetch the latest published state of each output directory (NATLANG_CW_RESUME=1, set by
# `cwctl.py launch --resume`), so a relaunched job continues from its last checkpoint at the same absolute path.
if [ "${NATLANG_CW_RESUME:-0}" = 1 ] && [ -n "${NATLANG_CW_BUCKET:-}" ]; then
  for dir in "${outputs[@]}"; do
    python3 "$BUNDLE" fetch --bucket "$NATLANG_CW_BUCKET" --missing-ok --root "$NATLANG_REPO" \
      --pointer-key "${NATLANG_CW_OUTPUT_PREFIX%/}/$(basename "$dir")/latest.json" || die "resume $dir"
  done
  log resumed
fi

publish() {
  [ -n "${NATLANG_CW_BUCKET:-}" ] || return 0
  for dir in "${outputs[@]}"; do
    [ -d "$dir" ] || continue
    python3 "$BUNDLE" publish "$dir" --bucket "$NATLANG_CW_BUCKET" --prefix "$NATLANG_CW_OUTPUT_PREFIX" \
      || log publish-failed "\"dir\":\"$dir\""
  done
}

# 6. the job; forward TERM so the trainer writes its final state, then publish outputs
cd "$NATLANG_REPO/${NATLANG_CW_WORKDIR:-}" || die workdir
export PYTHONPATH="$NATLANG_REPO/training/neuralese${PYTHONPATH:+:$PYTHONPATH}"
log run "\"command\":\"$*\""
"$@" &
child=$!
# Incremental publish beside the job: every new checkpoint reaches the bucket while the job runs, because lifetime
# expiry kills without a grace period (NATLANG_CW_PUBLISH_INTERVAL seconds between checks, default 60).
watcher=
if [ -n "${NATLANG_CW_BUCKET:-}" ] && [ ${#outputs[@]} -gt 0 ]; then
  python3 "$BUNDLE" watch "${outputs[@]}" --bucket "$NATLANG_CW_BUCKET" --prefix "$NATLANG_CW_OUTPUT_PREFIX" \
    --interval "${NATLANG_CW_PUBLISH_INTERVAL:-60}" > "$WORK/.coreweave-watch.log" 2>&1 &
  watcher=$!
fi
trap 'log term; kill -TERM $child 2>/dev/null' TERM INT
wait $child; status=$?
while kill -0 $child 2>/dev/null; do wait $child; status=$?; done  # a trapped TERM interrupts wait; keep waiting
log exited "\"status\":$status"
if [ -n "$watcher" ]; then kill -TERM $watcher 2>/dev/null; wait $watcher 2>/dev/null; fi
mkdir -p "$WORK/.coreweave"; echo "$status" > "$WORK/.coreweave/exit-status"
publish
if [ -n "${NATLANG_CW_BUCKET:-}" ]; then
  python3 "$BUNDLE" record-exit --bucket "$NATLANG_CW_BUCKET" --prefix "$NATLANG_CW_OUTPUT_PREFIX" --status "$status" \
    || log exit-record-failed
fi
log done "\"status\":$status"
# Always exit 0: a nonzero main-process exit makes the platform restart the container (docs sandbox-lifecycle.md),
# which would rerun the whole job. The job's status is in the log, $WORK/.coreweave/exit-status and <prefix>exit.json.
exit 0
