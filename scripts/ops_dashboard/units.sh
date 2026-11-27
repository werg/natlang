#!/bin/bash
# The dashboard's three long-running units, as memory-ledger services with fresh names (<role>-<epoch>):
#   ops-agent-N              `tendril agent run` in app/: the page's agent program (admits the owner's browsers, serves live data,
#                            applies edits; app/pages/ops/agent.mjs) with the Python data service as its child
#   ops-inbox-N              app/pages/ops/inbox-worker.mjs: receipts and inbox.jsonl for the questions from the page (needs the agent
#                            program: it attaches to it and exits when it goes away; `units.sh start` starts it again)
#   ops-dashboard-export-N   export.py: the exporter that writes the snapshot every 30 s
#   units.sh status                  the running units of each role
#   units.sh start [agent|inbox|export|all]   start a role that is not running (default all)
#   units.sh stop  [agent|inbox|export|all]   stop it (the inbox worker first)
#   units.sh restart [role|all]      stop, then start under fresh names
#   units.sh keep                    start the keeper unit ops-keeper-N (`units.sh watch`): every 30 s it starts what is missing, so a
#                                    crashed unit comes back. `units.sh stop` stops the keeper first; stop it before taking a unit down on purpose.
# Stopping the agent program does not lose anything: the page's data and the questions are in the journal.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../.." && pwd)
NODE_BIN=${NODE_BIN:-$HOME/.local/share/natlang-dev/node-v24.0.1-linux-arm64/bin}
LEDGER="$REPO/scripts/memory_ledger.py"
running() { systemctl --user list-units --state=active --no-legend --plain "$1-[0-9]*.service" 2>/dev/null | awk '{print $1}'; }
start_role() {
  case $1 in
    agent) "$LEDGER" run --unit "ops-agent-$(date +%s)" --budget-gb 1.5 --class service --priority medium --wait 120 --no-resumable \
             --workdir "$HERE/app" --env "PATH=$NODE_BIN:/usr/local/bin:/usr/bin:/bin" --env NODE_OPTIONS=--no-warnings --env OMP_NUM_THREADS=1 \
             -- "$NODE_BIN/node" "$HERE/app/node_modules/.bin/tendril" agent run ;;
    inbox) "$LEDGER" run --unit "ops-inbox-$(date +%s)" --budget-gb 0.5 --class service --priority medium --wait 120 --no-resumable \
             --workdir "$HERE/app" --env "PATH=$NODE_BIN:/usr/local/bin:/usr/bin:/bin" --env NODE_OPTIONS=--no-warnings \
             -- "$NODE_BIN/node" "$HERE/app/pages/ops/inbox-worker.mjs" ;;
    export) "$LEDGER" run --unit "ops-dashboard-export-$(date +%s)" --budget-gb 0.3 --class service --priority medium --wait 120 --no-resumable \
             -- /usr/bin/python3 "$HERE/export.py" ;;
    keeper) "$LEDGER" run --unit "ops-keeper-$(date +%s)" --budget-gb 0.1 --class service --priority medium --wait 120 --no-resumable \
             --env "PATH=$NODE_BIN:/usr/local/bin:/usr/bin:/bin" -- /bin/bash "$HERE/units.sh" watch ;;
    *) echo "unknown role $1" >&2; return 2 ;;
  esac
}
unit_of() { case $1 in agent) echo ops-agent;; inbox) echo ops-inbox;; export) echo ops-dashboard-export;; keeper) echo ops-keeper;; esac; }
roles() { case ${1:-all} in all) echo agent inbox export;; *) echo "$1";; esac; }
case ${1:-status} in
  status) for r in agent inbox export keeper; do echo "$r: $(running "$(unit_of $r)" | tr '\n' ' ')"; done ;;
  start) for r in $(roles "${2:-}"); do [ -n "$(running "$(unit_of $r)")" ] && echo "$r already running" || { start_role "$r" | tail -1; [ "$r" = agent ] && sleep 20; }; done ;;
  stop) for r in keeper inbox agent export; do case ${2:-all} in all|$r) for u in $(running "$(unit_of $r)"); do systemctl --user stop "$u" && echo "stopped $u"; done ;; esac; done ;;
  restart) "$0" stop "${2:-all}"; "$0" start "${2:-all}" ;;
  keep) [ -n "$(running ops-keeper)" ] && echo "keeper already running" || start_role keeper | tail -1 ;;
  watch) while true; do "$0" start all >/dev/null 2>&1; sleep 30; done ;;
  *) sed -n 2,16p "$0"; exit 2 ;;
esac
