#!/bin/sh
# The machine's specialization loop (plans/TRACE_SPECIALIZATION.md §7.7): one specializer pass (compile hot
# definitions, then shadow replays and audits) every INTERVAL seconds. Each pass starts a fresh process, so it always
# runs the newest code and data. Start it under the memory ledger, for example on the DGX:
#   python3 scripts/memory_ledger.py run --unit natlang-specializer --budget-gb 3 --class experiment \
#     --workdir /home/werg/natlang sh scripts/specializer-loop.sh --profile pi-executor
# Stop it with: systemctl --user stop natlang-specializer
INTERVAL=${NATLANG_SPECIALIZER_INTERVAL:-900}
LOG=${NATLANG_SPECIALIZER_LOG:-$HOME/.local/state/natlang/specializer.log}
mkdir -p "$(dirname "$LOG")"
cd "$(dirname "$0")/.." || exit 1
export PATH="$HOME/.local/bin:$PATH"
trap 'exit 0' TERM INT
while true; do
  node ts-host/bin/natlang.mjs specialize "$@" >> "$LOG" 2>&1
  sleep "$INTERVAL" &
  wait $!
done
