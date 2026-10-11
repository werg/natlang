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
REPO=/home/werg/natlang
# systemd units see /usr/bin/node (v18, no global crypto.randomUUID); use the toolchain node the checkout is built with.
NODE=${NODE:-/home/werg/.local/bin/node}
TS=$REPO/ts-host/scripts/inline-curriculum
RES=$REPO/data/external_pilot/jeff-test-tasks.jsonl
mkdir -p "$OUT/ir" "$OUT/turns" "$OUT/logs"
# Replay scratch lives under OUT/tmp (several GB per shard while a replay runs). A preempted run (SIGTERM) leaves
# no live replay behind, so the scratch of an earlier run is removed on start, and on TERM/INT/EXIT of this one.
rm -rf "$OUT/tmp"; mkdir -p "$OUT/tmp"
trap 'trap "" TERM INT; trap - EXIT; kill 0 2>/dev/null; wait; rm -rf "$OUT/tmp"' TERM INT EXIT
PAGE=$(cd "$REPO/ts-host" && "$NODE" -e "import('./dist/native/cutoff.js').then(m => console.log(typeof m.shownCharsFor === 'function' ? m.shownCharsFor(65536) : 'fixed-2000'))")
DIST=$(cat "$REPO/ts-host/dist/.build-commit" 2>/dev/null || echo "unknown (built before 2026-10-12 cutover)")
RUNTIME="{\"page_size\": \"$PAGE\", \"dist_commit\": \"$DIST\", \"context_tokens\": 65536}"
if [ -f "$OUT/runtime-page.json" ]; then
  [ "$(cat "$OUT/runtime-page.json")" = "$RUNTIME" ] || { echo "OUT was replayed with $(cat "$OUT/runtime-page.json"), runtime is now $RUNTIME: continue in a new corpus version (SKIP_DONE_IN)" >&2; exit 2; }
else
  echo "$RUNTIME" > "$OUT/runtime-page.json"
fi
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
  [ -f "$dst.summary.json" ] && return 0
  [ -n "${SKIP_DONE_IN:-}" ] && [ -f "$SKIP_DONE_IN/turns/$f/$n.summary.json" ] && return 0
  mkdir -p "$OUT/turns/$f"
  "$REPO/scripts/disk_guard.sh" "$OUT"  # pause while NVMe is short (coordinator 2026-10-12)
  tmp=$(mktemp -d "$OUT/tmp/replay.XXXXXX")
  if nice -n 15 "$NODE" --max-old-space-size=6144 "$TS/replay-port-record-calls.mjs" "$shard" "$tmp/x" > "$OUT/logs/$f.$n.replay.log" 2>&1; then
    zstd -q -T1 -6 --rm "$tmp/x.turns.jsonl" -o "$dst.turns.jsonl.zst.tmp" && mv "$dst.turns.jsonl.zst.tmp" "$dst.turns.jsonl.zst"
    zstd -q -T1 -6 --rm "$tmp/x.rows.jsonl" -o "$dst.rows.jsonl.zst.tmp" && mv "$dst.rows.jsonl.zst.tmp" "$dst.rows.jsonl.zst"
    python3 -c 'import json,sys; s=json.load(open(sys.argv[1])); s["runtime"]=json.loads(sys.argv[2]); json.dump(s, open(sys.argv[1], "w"), indent=1)' "$tmp/x.summary.json" "$RUNTIME"
    mv "$tmp/x.summary.json" "$dst.summary.json"
  else
    echo "replay failed: $shard (log $OUT/logs/$f.$n.replay.log)" >&2
  fi
  rm -rf "$tmp"
}
export -f build replay; export OUT IN TS RES CORPUS BUILD_ARGS NODE RUNTIME SKIP_DONE_IN REPO
families=$(ls "$IN" | sed -n 's/\.port-records\.jsonl$//p')
echo "$(date -Is) build $(echo $families | wc -w) families"
# Builds (I/O bound on the HDD families) run beside the replays: a family's shards replay as soon as its build is done.
printf '%s\n' $families | xargs -P 3 -I{} bash -c 'build {} || echo "build failed: {}" >&2' &
builder=$!
# return 0: under set -e/pipefail a last shard already done (the && chain fails) would end the whole driver.
pending() { for b in "$OUT"/ir/*/build.json; do [ -f "$b" ] || continue; d=$(dirname "$b"); f=$(basename "$d")
  for s in "$d"/*.ir.jsonl; do n=$(basename "$s" .ir.jsonl); [ -f "$s" ] && [ ! -f "$OUT/turns/$f/$n.summary.json" ] \
    && { [ -z "${SKIP_DONE_IN:-}" ] || [ ! -f "$SKIP_DONE_IN/turns/$f/$n.summary.json" ]; } && echo "$s"; done; done; return 0; }
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
done_=$(find "$OUT/turns" -name '*.summary.json' | wc -l); total=$(find "$OUT/ir" -name '*.ir.jsonl' | wc -l)
echo "$(date -Is) shards done $done_ / $total"
[ "$done_" -eq "$total" ]
