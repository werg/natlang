#!/bin/bash
# Port records -> natlang call programs -> native training turns (defects D15/D16), resumable.
#   port_record_calls.sh IN_DIR OUT_DIR [WORKERS] [CORPUS_ID]
# For every IN_DIR/<family>.port-records.jsonl: build IR shards once (build-port-record-calls.mjs; OUT_DIR/ir/<family>/
# build.json marks a finished build), then replay every shard not yet done (replay-port-record-calls.mjs) and keep
# turns and rows zstd-compressed under OUT_DIR/turns/<family>/NNNNN.{turns,rows}.jsonl.zst + .summary.json (the
# summary marks the shard done). Interrupt and rerun at any time; finished work is skipped.
# BUILD_ARGS passes builder options (large-evidence pass: BUILD_ARGS='--min-evidence 32000 --max-evidence 120000').
# Run under the DGX memory ledger:
#   python3 scripts/memory_ledger.py run --unit s1-natlang-<id> --budget-gb 16 -- scripts/neuralese_data/port_record_calls.sh IN OUT 6
set -euo pipefail
IN=$(realpath "$1"); OUT=$(realpath -m "$2"); WORKERS=${3:-6}; CORPUS=${4:-s1-full-final-20261003}
REPO=/home/werg/natlang
TS=$REPO/ts-host/scripts/inline-curriculum
RES=$REPO/data/external_pilot/jeff-test-tasks.jsonl
mkdir -p "$OUT/ir" "$OUT/turns" "$OUT/logs"
build() {
  f=$1
  [ -f "$OUT/ir/$f/build.json" ] && return 0
  rm -rf "$OUT/ir/$f"
  nice -n 15 node "$TS/build-port-record-calls.mjs" --input "$IN/$f.port-records.jsonl" --out "$OUT/ir" \
    --corpus-id "$CORPUS" --reservations "$RES" ${BUILD_ARGS:-} > "$OUT/logs/$f.build.log" 2>&1
}
replay() {
  shard=$1; f=$(basename "$(dirname "$shard")"); n=$(basename "$shard" .ir.jsonl)
  dst=$OUT/turns/$f/$n
  [ -f "$dst.summary.json" ] && return 0
  mkdir -p "$OUT/turns/$f"
  tmp=$(mktemp -d /home/werg/data/tmp-port-calls.XXXXXX)
  if nice -n 15 node --max-old-space-size=6144 "$TS/replay-port-record-calls.mjs" "$shard" "$tmp/x" > "$OUT/logs/$f.$n.replay.log" 2>&1; then
    zstd -q -T1 -6 --rm "$tmp/x.turns.jsonl" -o "$dst.turns.jsonl.zst.tmp" && mv "$dst.turns.jsonl.zst.tmp" "$dst.turns.jsonl.zst"
    zstd -q -T1 -6 --rm "$tmp/x.rows.jsonl" -o "$dst.rows.jsonl.zst.tmp" && mv "$dst.rows.jsonl.zst.tmp" "$dst.rows.jsonl.zst"
    mv "$tmp/x.summary.json" "$dst.summary.json"
  else
    echo "replay failed: $shard (log $OUT/logs/$f.$n.replay.log)" >&2
  fi
  rm -rf "$tmp"
}
export -f build replay; export OUT IN TS RES CORPUS BUILD_ARGS
families=$(ls "$IN" | sed -n 's/\.port-records\.jsonl$//p')
echo "$(date -Is) build $(echo $families | wc -w) families"
printf '%s\n' $families | xargs -P "$WORKERS" -I{} bash -c 'build {} || echo "build failed: {}" >&2'
echo "$(date -Is) replay"
find "$OUT/ir" -name '*.ir.jsonl' | sort | xargs -P "$WORKERS" -I{} bash -c 'replay {}'
done_=$(find "$OUT/turns" -name '*.summary.json' | wc -l); total=$(find "$OUT/ir" -name '*.ir.jsonl' | wc -l)
echo "$(date -Is) shards done $done_ / $total"
[ "$done_" -eq "$total" ]
