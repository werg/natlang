#!/bin/bash
# Publish the page: copy only the page files to a staging dir (make_fixture.py must not ship) and `tendril push` it.
# usage: scripts/ops_dashboard/publish.sh [message]    handle: ~/.local/share/tendriller/natlang-ops-v3/create.json (agent token, never printed)
set -euo pipefail
D=${TENDRIL_DIR:-$HOME/.local/share/tendriller/natlang-ops-v3}
SRC=$(cd "$(dirname "$0")/page" && pwd)
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
cp "$SRC"/{index.html,style.css,app.mjs,charts.mjs,model.mjs} "$STAGE"/
eval "$(python3 -I - "$D/create.json" <<'PY'
import json, sys, shlex
d = json.load(open(sys.argv[1]))
print('export TENDRIL_API=' + shlex.quote(d['api']) + ' TENDRIL_TOKEN=' + shlex.quote(d['token']))
PY
)"
export PATH="$HOME/.local/share/natlang-dev/node-v24.0.1-linux-arm64/bin:$PATH"
cd /tmp
# --no-wait: health stays pending until a visible tab loads the version, which can take as long as the owner needs to open the page
exec node "$D/tools/bin/tendril.mjs" push "$STAGE" --message "${1:-ops dashboard}" --no-wait
