#!/bin/bash
# The supervising agent's side of the ops page inbox. A question from the page is an `agentMessage` in the application (data =
# {kind, subject}); the inbox worker (unit ops-inbox, app/pages/ops/inbox-worker.mjs) gives it a receipt and mirrors every question
# with its answers to <dashboard>/inbox.jsonl, which this script reads.
#   inbox.sh                 print asks that are neither answered nor marked handled (JSON lines); counts as a read of the inbox
#   inbox.sh wait            block until such an ask exists, print it, exit 0 (for a background command in a session)
#   inbox.sh listener        print the age of the last read in seconds (the page shows it as "agent reading")
#   inbox.sh answer OP TEXT  answer ask OP in the page: a reply message through the running agent program (tendril attach)
#   inbox.sh handled OP      mark OP handled without an answer (e.g. an acknowledged alert)
# Every read (list, wait, answer) touches <dashboard>/inbox.listener; the agent program puts that mtime in its 10 s tick and the
# page says whether the supervising agent has looked lately. The questions themselves are kept whether or not anyone reads.
set -euo pipefail
DASH=$(python3 -c "import json,os;print(os.path.expanduser(json.load(open('$(dirname "$0")/config.json'))['data_dir'])+'/ops/dashboard')")
APP=$(cd "$(dirname "$0")/app" && pwd)
NODE=${NODE:-$HOME/.local/share/natlang-dev/node-v24.0.1-linux-arm64/bin/node}
touch_listener() { touch "$DASH/inbox.listener"; }
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
    if a['kind'] != 'ack_alert' and not a['answers'] and a['op'] not in handled:
        print(json.dumps(a, ensure_ascii=False))
PY
}
case "${1:-list}" in
  list) touch_listener; open_asks ;;
  wait) touch_listener
        until [ -n "$(open_asks)" ]; do sleep 10; touch_listener; done; open_asks ;;
  listener) python3 -c 'import os,sys,time;print(round(time.time()-os.path.getmtime(sys.argv[1])))' "$DASH/inbox.listener" 2>/dev/null || echo never ;;
  answer)
    [ $# -eq 3 ] || { echo "usage: $0 answer OP TEXT" >&2; exit 2; }
    touch_listener
    INPUT=$(python3 - "$DASH/inbox.jsonl" "$2" "$3" <<'PY'
import hashlib, json, sys
asks = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
a = next((x for x in asks if x['op'] == sys.argv[2]), None)
if a is None:
    sys.exit('no such ask in inbox.jsonl: ' + sys.argv[2])
print(json.dumps({'channel': 'chat', 'to': [a['author']], 'text': sys.argv[3][:8000], 'expectReply': False, 'hopCount': 1,
                  'inReplyTo': {'author': a['author'], 'operationId': a['op']}}))
PY
)
    OPID="answer-$(python3 -c 'import hashlib,sys;print(hashlib.sha256(sys.argv[1].encode()).hexdigest()[:40])' "$2")"
    cd "$APP" && NODE_NO_WARNINGS=1 "$NODE" node_modules/.bin/tendril --page ops command agentMessage "$INPUT" --operation "$OPID" --wait materialized
    echo "$2" >> "$DASH/inbox.handled" ;;
  handled) echo "$2" >> "$DASH/inbox.handled" ;;
  *) echo "usage: $0 [list|wait|listener|answer OP TEXT|handled OP]" >&2; exit 2 ;;
esac
