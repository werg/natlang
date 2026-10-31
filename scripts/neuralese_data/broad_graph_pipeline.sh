#!/bin/bash
# Broad Neuralese graphs: cases -> natlang IR -> custody -> reference replay -> e2e edges -> LFM3 chain graphs.
# plans/neuralese/NEURALESE_BROAD_GRAPHS_20261031.md. Resumable: every stage skips finished outputs.
#   broad_graph_pipeline.sh ROOT TS_BUILD CORPUS_ID [WORKERS]
# ROOT holds cases/cases.jsonl (broad_graph_cases.py build). TS_BUILD is a pinned checkout of a main commit with a
# built ts-host (ts-host/dist), recorded in ROOT/runtime.json; replay and conversion run on it.
# Run under the DGX memory ledger (low priority, resumable).
set -euo pipefail
ROOT=$(realpath "$1"); BUILD=$(realpath "$2"); CORPUS=$3; WORKERS=${4:-6}
REPO=${NATLANG_REPO:-/home/werg/natlang}
NODE=${NODE:-/home/werg/.local/bin/node}
PY=$REPO/.venv-neuralese/bin/python
export PYTHONPATH=$REPO/training/neuralese:$REPO
TS=$BUILD/ts-host/scripts/inline-curriculum
# The generic IR builder is taken from the canonical checkout and run inside the pinned tree (its lib imports the
# pinned dist); its exact bytes are recorded.
cp "$REPO/ts-host/scripts/inline-curriculum/build-broad-graph-calls.mjs" "$TS/build-broad-graph-calls.mjs"
RUNTIME="{\"build\": \"$BUILD\", \"dist_commit\": \"$(cat "$BUILD/ts-host/dist/.build-commit" 2>/dev/null || echo unknown)\", \"ir_builder_sha256\": \"$(sha256sum "$TS/build-broad-graph-calls.mjs" | cut -d' ' -f1)\"}"
if [ -f "$ROOT/runtime.json" ]; then
  [ "$(cat "$ROOT/runtime.json")" = "$RUNTIME" ] || { echo "ROOT was built with $(cat "$ROOT/runtime.json"); now $RUNTIME" >&2; exit 2; }
else echo "$RUNTIME" > "$ROOT/runtime.json"; fi
mkdir -p "$ROOT/logs" "$ROOT/turns" "$ROOT/tmp"
# 1. IR
if [ ! -f "$ROOT/ir-raw/.done" ]; then
  rm -rf "$ROOT/ir-raw"
  nice -n 15 "$NODE" "$TS/build-broad-graph-calls.mjs" --input "$ROOT/cases/cases.jsonl" --out "$ROOT/ir-raw" --shard-cases 100 \
    > "$ROOT/logs/ir.log" 2>&1
  touch "$ROOT/ir-raw/.done"
fi
# 2. protected-world custody on the exact program worlds
if [ ! -f "$ROOT/ir/custody.json" ]; then
  nice -n 15 "$PY" "$REPO/scripts/neuralese_data/broad_graph_cases.py" custody --ir "$ROOT/ir-raw" --out "$ROOT/ir" \
    --workers "$WORKERS" >> "$ROOT/logs/custody.log" 2>&1
fi
# 3. reference replay (current serving prompt, tool schemas, opening declarations) per shard
replay() {
  shard=$1; src=$(basename "$(dirname "$shard")"); n=$(basename "$shard" .ir.jsonl); dst=$ROOT/turns/$src/$n
  [ -f "$dst.summary.json" ] && return 0
  mkdir -p "$ROOT/turns/$src"; tmp=$(mktemp -d "$ROOT/tmp/replay.XXXXXX")
  if nice -n 15 "$NODE" --max-old-space-size=4096 "$TS/replay-port-record-calls.mjs" "$shard" "$tmp/x" > "$ROOT/logs/$src.$n.replay.log" 2>&1; then
    zstd -q -T1 -6 --rm "$tmp/x.turns.jsonl" -o "$dst.turns.jsonl.zst" && rm -f "$tmp/x.rows.jsonl"
    mv "$tmp/x.summary.json" "$dst.summary.json"
  else echo "replay failed: $shard" >&2; fi
  rm -rf "$tmp"
}
export -f replay; export ROOT TS NODE
find "$ROOT/ir" -name '*.ir.jsonl' -size +0 | sort | xargs -P "$WORKERS" -I{} bash -c 'replay {}'
# 4. e2e edges (convert on the pinned ts-host) and 5. chain graphs, per source
export E2E_MINE_TS_HOST=$BUILD/ts-host
mine() {
  dir=$1; src=$(basename "$dir")
  [ -f "$ROOT/graphs/$src/receipt.json" ] && return 0
  if [ ! -f "$ROOT/work/$src/sharded.json" ]; then
    "$PY" "$REPO/scripts/neuralese_data/e2e_mine.py" shard "$src" "$dir"*.turns.jsonl.zst --work "$ROOT/work" --shards \
      "$(( ($(ls "$dir"*.turns.jsonl.zst | wc -l) + 9) / 10 ))" > "$ROOT/logs/$src.shard.log" 2>&1
  fi
  nice -n 15 "$PY" "$REPO/scripts/neuralese_data/e2e_mine.py" convert --work "$ROOT/work" --source "$src" > "$ROOT/logs/$src.convert.log" 2>&1
  nice -n 15 "$PY" "$REPO/scripts/neuralese_data/build_chain_graphs.py" --work "$ROOT/work" --source "$src" \
    --out "$ROOT/graphs/$src" --corpus-id "$CORPUS" --prompt full --workers 2 > "$ROOT/logs/$src.chains.log" 2>&1 \
    || echo "chains failed: $src" >&2
}
export -f mine; export PY REPO CORPUS
ls -d "$ROOT"/turns/*/ | xargs -P "$(( (WORKERS + 1) / 2 ))" -I{} bash -c 'mine {}'
rm -rf "$ROOT/tmp"
echo "$(date -Is) done"
