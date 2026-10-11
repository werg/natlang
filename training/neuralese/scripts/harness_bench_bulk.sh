#!/bin/sh
# Bulk harness-bench records from the SWE-rebench OpenHands trajectories that harness-bench v4 did not cover
# (DATA_INVENTORY_20261011.md defect D13). Same stages as harness_bench_build.sh (surface -> prepare -> replay in pi's
# tools -> records with S1 placement), sharded by repository so several replays run at once without two of them
# cloning one repository into the shared bare-clone cache, and so a finished shard can be registered while the rest run.
# Usage: harness_bench_bulk.sh OUT [SHARDS=24] [PARALLEL=6] [OFFSET=1000]   (run it under scripts/memory_ledger.py run)
# OFFSET 1000: v4 is the first 1000 parquet rows (492 resolved). Resumable: every stage's output is written under a
# temporary name and renamed when complete (prepare per parquet row group); replay keeps what replayed.jsonl holds and appends the rest.
set -eu
OUT=$1; SHARDS=${2:-24}; PARALLEL=${3:-6}; OFFSET=${4:-1000}
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
PY="$ROOT/.venv-neuralese/bin/python"
CORPORA="$ROOT/data/neuralese/corpora"
WORKSPACE=/workspace/project
mkdir -p "$OUT/shards"
export PATH="$HOME/.local/bin:$PATH"
if [ ! -e "$OUT/surface.json" ]; then
  node "$ROOT/ts-host/bin/natlang.mjs" run "$ROOT/applications/pi" -- surface --cwd "$WORKSPACE" --companion > "$OUT/surface.json.tmp"
  mv "$OUT/surface.json.tmp" "$OUT/surface.json"
fi
cd "$ROOT/training/neuralese"
# Prepare one parquet row group (4096 rows) at a time, so a preempted run resumes at the next unfinished group.
TOTAL=$("$PY" -c "import pyarrow.parquet as pq,sys; print(pq.ParquetFile(sys.argv[1]).metadata.num_rows)" "$CORPORA/nebius-swe-rebench-openhands-trajectories-20261009-v1/trajectories.parquet")
mkdir -p "$OUT/prepared"
START=$OFFSET
while [ "$START" -lt "$TOTAL" ]; do
  END=$(( (START / 4096 + 1) * 4096 ))
  PART="$OUT/prepared/$(printf %06d "$START").jsonl"
  if [ ! -e "$PART" ]; then
    nice -n 10 "$PY" -m natlang_neuralese.harness_bench.prepare --trajectories "$CORPORA/nebius-swe-rebench-openhands-trajectories-20261009-v1/trajectories.parquet" \
      --tasks "$CORPORA/nebius-swe-rebench-tasks-20261009-v1" --workspace "$WORKSPACE" --resolved-only \
      --offset "$START" --limit $((END - START)) --out "$PART.tmp" > "$PART.summary.json"
    mv "$PART.tmp" "$PART"
  fi
  START=$END
done
if [ ! -e "$OUT/shards/.split" ]; then
  "$PY" - "$OUT/prepared" "$OUT/shards" "$SHARDS" <<'EOF'
import hashlib, json, sys
from pathlib import Path
src, out, n = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
parts = sorted(src.glob('*.jsonl'))
handles = []
for i in range(n):
    (out / f'{i:02d}').mkdir(parents=True, exist_ok=True)
    handles.append(open(out / f'{i:02d}' / 'prepared.jsonl', 'w'))
for part in parts:
    with open(part) as lines:
        for line in lines:
            if line.strip():
                repo = json.loads(line)['repo']
                handles[int(hashlib.sha256(repo.encode()).hexdigest(), 16) % n].write(line)
for h in handles:
    h.close()
EOF
  touch "$OUT/shards/.split"
fi
# The shards hold every prepared trajectory now; the row-group parts are duplicates (their summaries stay).
rm -f "$OUT"/prepared/*.jsonl
# One shard: replay (resumable), then its records once replay finished (the records stage is per repository, so a
# shard of whole repositories places exactly as the whole would).
cat > "$OUT/shard.sh" <<EOF
#!/bin/sh
set -eu
D=\$1
[ -e "\$D/summary.json" ] && exit 0
"$ROOT/scripts/disk_guard.sh" "$OUT"  # pause while the NVMe is short (coordinator 2026-10-12)
nice -n 10 node "$ROOT/ts-host/bin/natlang.mjs" run "$ROOT/applications/pi" -- replay --in "\$D/prepared.jsonl" --out "\$D/replayed.jsonl" 2>> "\$D/replay.log"
cd "$ROOT/training/neuralese"
nice -n 10 "$PY" -m natlang_neuralese.harness_bench.records --replayed "\$D/replayed.jsonl" --surface "$OUT/surface.json" \\
  --corpus nebius-swe-rebench-openhands-trajectories-20261009-v1 --out "\$D/records.tmp"
for f in "\$D"/records.tmp/*; do mv "\$f" "\$D/"; done
rmdir "\$D/records.tmp"
echo "shard \$D done \$(date -u +%FT%TZ)"
EOF
chmod +x "$OUT/shard.sh"
# PARALLEL replays at once (each mostly waits on git fetches); the records stage after each is CPU and disk bound.
ls -d "$OUT"/shards/[0-9]* | xargs -P "$PARALLEL" -n 1 "$OUT/shard.sh"
echo "all shards done $(date -u +%FT%TZ)"
