#!/bin/sh
# usage: start-server.sh UNIT PORT SLOTS  (retries ledger admission until the server is healthy)
UNIT=$1; PORT=$2; NP=$3
W=/home/werg/natlang/.claude/worktrees/agent-a3e5fda6488bf57f1
B=/home/werg/llama.cpp-neuralese/build-cuda/bin
M=$(cat /tmp/bench-model)
CTX=$((8192 * NP))
while ! curl -sf localhost:$PORT/health >/dev/null; do
  python3 $W/scripts/memory_ledger.py run --unit $UNIT --budget-gb 12 --class experiment --wait 900 --workdir $W -- \
    env LD_LIBRARY_PATH=$B $B/llama-server -m $M --host 127.0.0.1 --port $PORT --parallel $NP -c $CTX -ngl 99 \
    --cache-ram 256 --no-webui --jinja --chat-template-file $W/models/templates/LFM2.5-350M.jinja >> /tmp/bench/ledger-$UNIT.log 2>&1
  for i in $(seq 1 100); do systemctl --user is-active --quiet $UNIT.service || [ $i -lt 4 ] || break; curl -sf localhost:$PORT/health >/dev/null && break; sleep 3; done
done
echo up
