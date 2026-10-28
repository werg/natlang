#!/bin/bash
# Development-packet monitoring of a running LFM3 DiLoCo stage, in its own GPU sandbox next to the coordinator
# (plans/neuralese/LFM3_SFT_RUN_20261028.md). Global versions never pass through DGX: DGX uploads run at
# 0.2-0.5 MB/s and its GPU memory is contended. Monitoring only, never a gate (LFM3_COMMITTED_DECISIONS #15).
#
#   lfm3_dev_eval_global.sh EVERY_N [PARENT] [BASE]
#
# Launched with `diloco.py attach-eval RUN_ID JOB.json BUNDLE.json`, which supplies NATLANG_DILOCO_URL and the
# learners' secrets (diloco-token, diloco-gateway under $NATLANG_CW_SECRETS). PARENT (the stage's parent checkpoint,
# for config) and BASE default to the seed snapshot's /work/parent/checkpoint.pt and pinned LFM3 base.
#
# Loop: the global round is the minimum committed fragment version (/v1/status "versions"). The first round seen
# (round 0 = the parent, the base reference under the current oracles) and then every EVERY_N rounds:
#   1. native_diloco export-global --coordinator-url: model-only global and EMA checkpoints over the API;
#   2. export.native_gguf -> BF16 GGUF with the pinned fork's converter snapshot (bundled under
#      .development-source-snapshots; vendor/llama.cpp is a DGX-only symlink);
#   3. lfm3_serve.sh eval on dev packet v2 (scripts/coreweave/jobs/lfm3/dev-packet-spec.json) through the prebuilt
#      llama-cpp-neuralese-sm120 server.
# Results: /work/dev-evals/r<round>/{global,ema}/ (probe/, probe.log, timings.json, gguf manifest.json), plus
# /work/dev-evals/index.jsonl (one line per evaluated version). The checkpoints and GGUFs are deleted after each eval.
# FP8/NVFP4: llama.cpp loads neither compressed-tensors FP8_BLOCK nor ModelOpt NVFP4, so those columns are recorded as
# unsupported here (the QAT points are exported with export/lfm3_quantized for vLLM/SGLang serving).
# Stops on /work/release or when the coordinator reports the stage finished.
set -uo pipefail
EVERY=${1:?EVERY_N}
PARENT=${2:-/work/parent/checkpoint.pt}
BASE=${3:-/work/hf/models--LiquidAI--LFM3-1.7B/snapshots/40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd}
URL=${NATLANG_DILOCO_URL:?attach-eval sets NATLANG_DILOCO_URL}
REPO=${NATLANG_REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
CONVERTER=${LFM3_GGUF_CONVERTER:-$(ls -d "$REPO"/.development-source-snapshots/llama.cpp-* 2>/dev/null | head -1)}
SPEC=${LFM3_DEV_SPEC:-scripts/coreweave/jobs/lfm3/dev-packet-spec.json}
ROOT=/work/dev-evals
mkdir -p "$ROOT"
cd "$REPO"
{ echo "export PYTHONPATH='${PYTHONPATH:-}'"; echo "export PATH='$PATH'"; } > /work/.hold-env
[ -n "$CONVERTER" ] && [ -f "$CONVERTER/convert_hf_to_gguf.py" ] || { echo "no converter snapshot ($CONVERTER)" >&2; exit 2; }
log() { echo "{\"at\":\"$(date -u +%FT%TZ)\",$1}" | tee -a "$ROOT/driver.jsonl"; }

status() {
  python3 - "$URL" <<'PY'
import json, sys
from natlang_neuralese.train.diloco.coordinator import TOKEN_ENV, TOKEN_SECRET, credential
from natlang_neuralese.train.diloco.worker import GATEWAY_SECRET, GATEWAY_TOKEN_ENV, Transport
t = Transport(sys.argv[1], credential(TOKEN_ENV, TOKEN_SECRET), 120,
              gateway=credential(GATEWAY_TOKEN_ENV, GATEWAY_SECRET))
s = t.json('GET', '/v1/status')
print(json.dumps({'round': min(s['versions']) if s.get('versions') else -1, 'finished': bool(s.get('finished')),
                  'merges': s.get('merges')}))
PY
}

last=-1
while [ ! -f /work/release ]; do
  st=$(status) || { log "\"event\":\"status-failed\""; sleep 60; continue; }
  round=$(python3 -c "import json,sys;print(json.loads(sys.argv[1])['round'])" "$st")
  finished=$(python3 -c "import json,sys;print(int(json.loads(sys.argv[1])['finished']))" "$st")
  if [ "$round" -ge 0 ] && { [ "$last" -lt 0 ] || [ "$round" -ge $((last + EVERY)) ] || [ "$finished" = 1 -a "$round" -gt "$last" ]; }; then
    dir="$ROOT/r$round"; mkdir -p "$dir"
    log "\"event\":\"export\",\"round\":$round"
    if python -m natlang_neuralese.train.native_diloco export-global --coordinator-url "$URL" --parent "$PARENT" \
         --base "$BASE" --out "$dir/ckpt/global" --ema-out "$dir/ckpt/ema" --workdir "$dir/fetch" > "$dir/export.log" 2>&1; then
      for kind in global ema; do
        [ -f "$dir/ckpt/$kind/checkpoint.pt" ] || continue
        if python -m natlang_neuralese.export.native_gguf "$dir/ckpt/$kind/checkpoint.pt" "$dir/$kind/gguf" \
             --outtype bf16 --llama-cpp "$CONVERTER" > "$dir/$kind/gguf.log" 2>&1; then
          gguf="$dir/$kind/gguf/model-bf16.gguf"; sha=$(sha256sum "$gguf" | cut -d' ' -f1)
          LFM3_GGUF="$gguf" bash scripts/coreweave/lfm3_serve.sh eval "$dir/$kind" "$SPEC" unused "$sha" \
            > "$dir/$kind/serve.log" 2>&1
          rc=$?
          log "\"event\":\"evaluated\",\"round\":$round,\"kind\":\"$kind\",\"exit\":$rc,\"gguf_sha256\":\"$sha\",\"fp8\":\"unsupported-by-llama.cpp\",\"nvfp4\":\"unsupported-by-llama.cpp\""
          python3 - "$dir/$kind" "$round" "$kind" "$sha" "$rc" >> "$ROOT/index.jsonl" <<'PY'
import glob, json, os, sys
out, rnd, kind, sha, rc = sys.argv[1:]
summary = None
for path in sorted(glob.glob(os.path.join(out, 'probe', '*.json'))):
    try:
        value = json.load(open(path))
    except ValueError:
        continue
    if isinstance(value, dict) and ('successes' in value or 'summary' in value):
        summary = value.get('summary', value)
print(json.dumps({'round': int(rnd), 'kind': kind, 'gguf_sha256': sha, 'exit': int(rc), 'summary': summary,
                  'precisions': {'bf16': 'evaluated', 'fp8': 'unsupported-by-llama.cpp', 'nvfp4': 'unsupported-by-llama.cpp'}}))
PY
          rm -f "$gguf"
        else
          log "\"event\":\"gguf-failed\",\"round\":$round,\"kind\":\"$kind\""
        fi
      done
    else
      log "\"event\":\"export-failed\",\"round\":$round"
    fi
    rm -rf "$dir/ckpt" "$dir/fetch"
    last=$round
  fi
  [ "$finished" = 1 ] && [ "$round" -le "$last" ] && { log "\"event\":\"stage-finished\",\"round\":$round"; break; }
  sleep 120
done
