#!/bin/sh
# Reference Neuralese server on LFM2.5-350M (CPU) at port 18743, admitted through the ledger.
W=/home/werg/natlang/.claude/worktrees/agent-a3e5fda6488bf57f1
while ! curl -sf localhost:18743/health >/dev/null 2>&1 && ! curl -sf localhost:18743/v1/neuralese/info >/dev/null 2>&1; do
  python3 $W/scripts/memory_ledger.py run --unit natlang-bench-ref --budget-gb 12 --class experiment --wait 900 --workdir $W -- \
    sh -c "cd $W/training/neuralese && HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=$W/training/neuralese exec /home/werg/natlang/.venv-neuralese/bin/python -m natlang_neuralese.serve --port 18743 --max-block 4 --threads 12 --device cpu" >> /tmp/bench/ledger-ref.log 2>&1
  for i in $(seq 1 120); do systemctl --user is-active --quiet natlang-bench-ref.service || [ $i -lt 4 ] || break; curl -sf localhost:18743/v1/neuralese/info >/dev/null 2>&1 && break; sleep 3; done
done
echo up
