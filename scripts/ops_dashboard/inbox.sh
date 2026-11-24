#!/bin/bash
# The supervising agent's side of the ops page inbox. Questions, stop requests and alert acknowledgements are signed `ask`
# operations in the application; the native participant (app/ops-peer.mjs) mirrors them to <dashboard>/inbox.jsonl.
#   inbox.sh                 print asks that are neither answered nor marked handled (JSON lines)
#   inbox.sh wait            block until such an ask exists, print it, exit 0 (run as an uncapped background command). It touches
#                            <dashboard>/inbox.listener every 10 s; the page shows that heartbeat so the owner knows someone is listening.
#                            Re-arm it after every ask AND after every restart of the ops participant (ops-peer-N).
#   inbox.sh listener        print the age of the listener heartbeat in seconds
#   inbox.sh answer OP TEXT  answer ask OP in the page (declared command `answer`, applied by the running participant)
#   inbox.sh handled OP      mark OP handled without an answer (e.g. an acknowledged alert)
set -euo pipefail
DASH=$(python3 -c "import json,os;print(os.path.expanduser(json.load(open('$(dirname "$0")/config.json'))['data_dir'])+'/ops/dashboard')")
APP=$(cd "$(dirname "$0")/app" && pwd)
NODE=${NODE:-$HOME/.local/share/natlang-dev/node-v24.0.1-linux-arm64/bin/node}
open_asks() {
  python3 - "$DASH/inbox.jsonl" "$DASH/inbox.handled" <<'PY'
import json, sys
try:
    handled = set(open(sys.argv[2]).read().split())
except OSError:
    handled = set()
try:
    lines = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
except OSError:
    lines = []
for a in lines:
    if not a['answers'] and a['op'] not in handled and a['op'][-128:] not in handled:
        print(json.dumps(a, ensure_ascii=False))
PY
}
case "${1:-list}" in
  list) open_asks ;;
  wait) touch "$DASH/inbox.listener"
        until [ -n "$(open_asks)" ]; do sleep 10; touch "$DASH/inbox.listener"; done; open_asks ;;
  listener) python3 -c 'import os,sys,time;print(round(time.time()-os.path.getmtime(sys.argv[1])))' "$DASH/inbox.listener" 2>/dev/null || echo never ;;
  answer)
    [ $# -eq 3 ] || { echo "usage: $0 answer OP TEXT" >&2; exit 2; }
    REF=$(python3 -c 'import sys;print(sys.argv[1][-128:])' "$2")
    INPUT=$(python3 -c 'import json,sys;print(json.dumps({"ask":sys.argv[1],"text":sys.argv[2][:4000]}))' "$REF" "$3")
    cd "$APP" && "$NODE" node_modules/.bin/tendril --session "$APP/session.mjs" command answer "$INPUT" --operation "answer-$(python3 -c 'import sys;print(sys.argv[1][-100:])' "$REF")" 2>/dev/null
    echo "$2" >> "$DASH/inbox.handled" ;;
  handled) echo "$2" >> "$DASH/inbox.handled" ;;
  *) echo "usage: $0 [list|wait|listener|answer OP TEXT|handled OP]" >&2; exit 2 ;;
esac
