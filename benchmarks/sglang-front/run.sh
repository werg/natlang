#!/bin/bash
# SGLang front throughput without a GPU: the front as serve_native_checkpoint.py builds it (front.py), bound to a mock
# SGLang that replays recorded collection turns (mock_sglang.py, RATE tokens/s per stream, 8-token chunks), and
# STREAMS streaming chat clients replaying recorded episodes (client.py). Prints steady-state tok/s (from the mock's
# served-token counter, 60 s .. SECONDS-30 s), the front's CPU cores and CPU ms per token. PROFILE='--gil --nonblocking'
# runs the front under py-spy (raw stacks in OUT/prof.txt; py-spy must be the front's parent where ptrace is scoped).
#   python extract.py COLLECT_DIR episodes.json 400       # recorded turns of a collection run
#   BENCH_CHECKPOINT=... BENCH_EXPORT=... [BENCH_BASE=...] run.sh REPO LABEL RATE STREAMS SECONDS
# Run it under the memory ledger (DGX) or scripts/colab/admission.py --preemptible (Colab): ~8 GB peak while loading.
set -u
REPO=$1; LABEL=$2; RATE=$3; STREAMS=$4; SECONDS_=$5
B=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-$REPO/.venv-neuralese/bin/python}
EPISODES=${EPISODES:-$B/episodes.json}
FP=${FRONT_PORT:-18990}; MP=${MOCK_PORT:-18991}
OUT=${OUT_ROOT:-$B}/out-$LABEL; mkdir -p $OUT
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1
$PY $B/mock_sglang.py $EPISODES $MP $RATE > $OUT/mock.log 2>&1 & MOCK=$!
WRAP=(); [ -n "${PROFILE:-}" ] && WRAP=(${PYSPY:-py-spy} record --rate 25 $PROFILE --format raw -o $OUT/prof.txt --)
NATLANG_REPO=$REPO PYTHONPATH=$REPO/training/neuralese:$REPO "${WRAP[@]}" $PY $B/front.py \
  "$BENCH_CHECKPOINT" "$BENCH_EXPORT" http://127.0.0.1:$MP $FP > $OUT/front.log 2>&1 & FRONT=$!
for i in $(seq 600); do grep -q bench_front_serving $OUT/front.log && break; kill -0 $FRONT || { echo front died; tail $OUT/front.log; kill $MOCK; exit 1; }; sleep 1; done
echo "front up after $i s"
FPID=$FRONT; [ -n "${PROFILE:-}" ] && FPID=$(pgrep -P $FRONT | head -1)
PROCS=4; THREADS=$((STREAMS / PROCS))
$PY $B/client.py $EPISODES $FP $PROCS $THREADS $SECONDS_ > $OUT/client.json 2> $OUT/client.err & CLIENT=$!
cpu(){ awk '{print $14+$15}' /proc/$1/stat; }
tok(){ curl -s http://127.0.0.1:$MP/stats | python3 -c 'import json,sys;print(json.load(sys.stdin)["tokens"])'; }
sleep 60
c0=$(cpu $FPID); m0=$(cpu $MOCK); k0=$(tok); t0=$(date +%s.%N)
sleep $((SECONDS_ - 90))
c1=$(cpu $FPID); m1=$(cpu $MOCK); k1=$(tok); t1=$(date +%s.%N)
for i in 1 2 3; do curl -s -o /dev/null -w "metrics_latency %{time_total}\n" --max-time 60 http://127.0.0.1:$FP/metrics; done
python3 - <<PY
dt=$t1-$t0
print({'label':'$LABEL','rate_per_stream':$RATE,'streams':$STREAMS,'steady_tok_s':round(($k1-$k0)/dt,1),
       'front_cores':round(($c1-$c0)/100/dt,2),'mock_cores':round(($m1-$m0)/100/dt,2),
       'front_cpu_ms_per_tok':round(($c1-$c0)*10/max(1,$k1-$k0),3)})
PY
timeout 240 tail --pid=$CLIENT -f /dev/null; kill $CLIENT 2>/dev/null
cat $OUT/client.json; curl -s http://127.0.0.1:$MP/stats; echo
curl -s --max-time 60 http://127.0.0.1:$FP/metrics | grep -E "fallback_requests|sglang_requests|generated_tokens"
kill -INT $FRONT; timeout 60 tail --pid=$FRONT -f /dev/null; kill -9 $FPID $FRONT 2>/dev/null; kill -9 $MOCK; timeout 30 tail --pid=$FPID -f /dev/null
exit 0
