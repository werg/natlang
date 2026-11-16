#!/bin/bash
# Block until the dashboard page has an unhandled waking event, print the batch (JSON) and exit 0. Run as an uncapped background command;
# the exit is the notice. Then: `wait_event.sh pending` (events after the handled cursor), do the work, `wait_event.sh ack SEQ`.
#   wait_event.sh            block for the next waking event (returns at once when one is already unhandled)
#   wait_event.sh pending    print unhandled events (probes are auto-acknowledged)   [--pretty]
#   wait_event.sh ack SEQ    mark events up to SEQ handled (after replying / acting)
# Handle (agent token, never printed): ~/.local/share/tendriller/natlang-ops-v3/create.json; handled cursor: create.json.events.json
set -euo pipefail
D=${TENDRIL_DIR:-$HOME/.local/share/tendriller/natlang-ops-v3}
NODE=$HOME/.local/share/natlang-dev/node-v24.0.1-linux-arm64/bin/node
cd /tmp
case "${1:-wait}" in
  pending) shift; exec "$NODE" "$D/tools/bin/tendril-events.mjs" "$D/create.json" pending "$@" ;;
  ack) shift; exec "$NODE" "$D/tools/bin/tendril-events.mjs" "$D/create.json" ack "$@" ;;
  wait)
    eval "$(python3 -I - "$D/create.json" "$D/create.json.events.json" <<'PY'
import json, sys, shlex, os
d = json.load(open(sys.argv[1]))
try:
    after = json.load(open(sys.argv[2])).get('handled_cursor', 0)
except OSError:
    after = 0
print('export TENDRIL_API=%s TENDRIL_TOKEN=%s TENDRIL_AFTER=%d' % (shlex.quote(d['api']), shlex.quote(d['token']), after))
PY
)"
    exec "$NODE" "$D/tools/bin/tendril-wait.mjs" ;;
  *) echo "usage: $0 [wait|pending|ack SEQ]" >&2; exit 2 ;;
esac
