#!/bin/bash
# Port records -> natlang call programs -> native training turns (defects D15/D16), resumable.
#   port_record_calls.sh IN_DIR OUT_DIR [WORKERS] [CORPUS_ID]
# For every IN_DIR/<family>.port-records.jsonl: build IR shards once (build-port-record-calls.mjs; OUT_DIR/ir/<family>/
# build.json marks a finished build), then replay every shard not yet done (replay-port-record-calls.mjs) and keep
# turns and rows zstd-compressed under OUT_DIR/turns/<family>/NNNNN.{turns,rows}.jsonl.zst + .summary.json (the
# summary marks the shard done). Interrupt and rerun at any time; finished work is skipped.
# SKIP_DONE_IN=DIR skips shards already replayed in an earlier version's OUT (a runtime cutover continues there).
# Page-size provenance: OUT/runtime-page.json records the tool-output page size of the ts-host build the replays use
# (fixed-2000 before 8267b177; an eighth of the 65,536-token budget after) and that build's commit, and every shard
# summary carries the same fields. A driver started on a different build than OUT records refuses to run: one corpus
# version, one page size.
# BUILD_ARGS passes builder options (large-evidence pass: BUILD_ARGS='--min-evidence 32000 --max-evidence 120000').
# Run under the DGX memory ledger:
#   python3 scripts/memory_ledger.py run --unit s1-natlang-<id> --budget-gb 16 -- scripts/neuralese_data/port_record_calls.sh IN OUT 6
set -euo pipefail
IN=$(realpath "$1"); OUT=$(realpath -m "$2"); WORKERS=${3:-6}; CORPUS=${4:-s1-full-final-20261003}
REPO=${NATLANG_REPO:-/home/werg/natlang}
# systemd units see /usr/bin/node (v18, no global crypto.randomUUID); use the toolchain node the checkout is built with.
NODE=${NODE:-/home/werg/.local/bin/node}
TS=$REPO/ts-host/scripts/inline-curriculum
RES=${PORT_RESERVATIONS:-$REPO/data/external_pilot/jeff-test-tasks.jsonl}
mkdir -p "$OUT/ir" "$OUT/turns" "$OUT/logs"
if [ -n "${SKIP_DONE_IN:-}" ]; then
  [ -n "${SKIP_DONE_RECEIPT:-}" ] || { echo "predecessor receipt required" >&2; exit 2; }
  python3 "$(dirname "${BASH_SOURCE[0]}")/port_predecessor_receipt.py" verify "$SKIP_DONE_IN" "$SKIP_DONE_RECEIPT"
  python3 - "$SKIP_DONE_IN" "$SKIP_DONE_RECEIPT" "$OUT/predecessor-binding.json" <<'PYCODE'
import hashlib,json,sys
from pathlib import Path
p=Path(sys.argv[2]).resolve(); Path(sys.argv[3]).write_text(json.dumps({'schema':'natlang.port-predecessor-binding/1','path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'source':str(Path(sys.argv[1]).resolve())},sort_keys=True)+'\n')
PYCODE
fi
# Cleanup and shard publication require one live driver for this output tree.
# Children inherit the descriptor, so a replacement waits for surviving work too.
exec 9>"$OUT/.pipeline.lock"
flock -n 9 || { echo "another port-record driver owns $OUT" >&2; exit 75; }
# Replay scratch lives under OUT/tmp (several GB per shard while a replay runs). A preempted run (SIGTERM) leaves
# no live replay behind, so the scratch of an earlier run is removed on start, and on TERM/INT/EXIT of this one.
rm -rf "$OUT/tmp"; mkdir -p "$OUT/tmp"
trap 'trap "" TERM INT; trap - EXIT; kill 0 2>/dev/null; wait; rm -rf "$OUT/tmp"' TERM INT EXIT
# The stamp alone is not an identity: a rebuild can leave it stale. Bind the
# actual frozen dist, dependency and script contents before admitting work.
RUNTIME=$(PYTHONPATH="$REPO/training/neuralese${PYTHONPATH:+:$PYTHONPATH}" python3 "$(dirname "${BASH_SOURCE[0]}")/port_runtime_identity.py" "$REPO" "$NODE" "$RES")
if [ -f "$OUT/runtime-page.json" ]; then
  [ "$(cat "$OUT/runtime-page.json")" = "$RUNTIME" ] || { echo "runtime content changed: use a new corpus output with SKIP_DONE_IN" >&2; exit 2; }
else
  printf '%s\n' "$RUNTIME" > "$OUT/runtime-page.json"
fi

shard_complete() {
  [ -f "$1.summary.json" ] && [ -s "$1.turns.jsonl.zst" ] && [ -s "$1.rows.jsonl.zst" ]
}
build() {
  f=$1
  [ -f "$OUT/ir/$f/build.json" ] && return 0
  rm -rf "$OUT/ir/$f"
  nice -n 15 "$NODE" "$TS/build-port-record-calls.mjs" --input "$IN/$f.port-records.jsonl" --out "$OUT/ir" \
    --corpus-id "$CORPUS" --reservations "$RES" ${BUILD_ARGS:-} > "$OUT/logs/$f.build.log" 2>&1
}
replay() {
  shard=$1; f=$(basename "$(dirname "$shard")"); n=$(basename "$shard" .ir.jsonl)
  dst=$OUT/turns/$f/$n
  shard_complete "$dst" && return 0
  [ -n "${SKIP_DONE_IN:-}" ] && shard_complete "$SKIP_DONE_IN/turns/$f/$n" && cmp -s "$shard" "$SKIP_DONE_IN/ir/$f/$n.ir.jsonl" && return 0
  mkdir -p "$OUT/turns/$f"
  # A historical failed publication may have a summary but a missing stream.
  # Invalidate that false completion before attempting its replacement.
  rm -f "$dst.summary.json"
  "$REPO/scripts/disk_guard.sh" "$OUT"  # pause while NVMe is short (coordinator 2026-10-12)
  tmp=$(mktemp -d "$OUT/tmp/replay.XXXXXX")
  if nice -n 15 "$NODE" --max-old-space-size=6144 "$TS/replay-port-record-calls.mjs" "$shard" "$tmp/x" > "$OUT/logs/$f.$n.replay.log" 2>&1; then
    # Interrupted compression can leave .tmp files. Only this locked owner may
    # replace them, and the summary is committed only after BOTH streams succeed.
    if zstd -q -f -T1 -6 --rm "$tmp/x.turns.jsonl" -o "$dst.turns.jsonl.zst.tmp" &&
       mv "$dst.turns.jsonl.zst.tmp" "$dst.turns.jsonl.zst" &&
       zstd -q -f -T1 -6 --rm "$tmp/x.rows.jsonl" -o "$dst.rows.jsonl.zst.tmp" &&
       mv "$dst.rows.jsonl.zst.tmp" "$dst.rows.jsonl.zst" &&
       python3 -c 'import json,sys; s=json.load(open(sys.argv[1])); s["runtime"]=json.loads(sys.argv[2]); json.dump(s, open(sys.argv[1], "w"), indent=1)' "$tmp/x.summary.json" "$RUNTIME" &&
       mv "$tmp/x.summary.json" "$dst.summary.json"; then
      :
    else
      echo "shard publication failed: $shard; no completion summary committed" >&2
      rm -rf "$tmp"
      return 1
    fi
  else
    echo "replay failed: $shard (log $OUT/logs/$f.$n.replay.log)" >&2
    rm -rf "$tmp"
    return 1
  fi
  rm -rf "$tmp"
}
export -f shard_complete build replay; export OUT IN TS RES CORPUS BUILD_ARGS NODE RUNTIME SKIP_DONE_IN REPO
families=$(ls "$IN" | sed -n 's/\.port-records\.jsonl$//p')
echo "$(date -Is) build $(echo $families | wc -w) families"
# Builds (I/O bound on the HDD families) run beside the replays: a family's shards replay as soon as its build is done.
printf '%s\n' $families | xargs -P "${BUILD_WORKERS:-3}" -I{} bash -c 'build {} || echo "build failed: {}" >&2' &
builder=$!
# return 0: under set -e/pipefail a last shard already done (the && chain fails) would end the whole driver.
pending() { for b in "$OUT"/ir/*/build.json; do [ -f "$b" ] || continue; d=$(dirname "$b"); f=$(basename "$d")
  for s in "$d"/*.ir.jsonl; do n=$(basename "$s" .ir.jsonl); [ -f "$s" ] && ! shard_complete "$OUT/turns/$f/$n" \
    && { [ -z "${SKIP_DONE_IN:-}" ] || ! shard_complete "$SKIP_DONE_IN/turns/$f/$n" || ! cmp -s "$s" "$SKIP_DONE_IN/ir/$f/$n.ir.jsonl"; } && echo "$s"; done; done; return 0; }
while :; do
  building=0; kill -0 $builder 2>/dev/null && building=1
  work=$(pending | sort)
  if [ -n "$work" ]; then printf '%s\n' $work | xargs -P "$WORKERS" -I{} bash -c 'replay {}'
  elif [ $building = 1 ]; then sleep 30
  else break; fi
  # a shard whose replay failed stays pending: stop retrying once nothing is building and a pass made no progress
  [ $building = 0 ] && [ "$(pending | sort)" = "$work" ] && break
done
wait $builder || true
done_=$(for s in "$OUT"/ir/*/*.ir.jsonl; do
  [ -f "$s" ] || continue
  f=$(basename "$(dirname "$s")"); n=$(basename "$s" .ir.jsonl)
  if shard_complete "$OUT/turns/$f/$n" ||
     { [ -n "${SKIP_DONE_IN:-}" ] && shard_complete "$SKIP_DONE_IN/turns/$f/$n"; }; then echo "$s"; fi
done | wc -l); total=$(find "$OUT/ir" -name '*.ir.jsonl' | wc -l)
echo "$(date -Is) shards done $done_ / $total"
[ "$done_" -eq "$total" ]
