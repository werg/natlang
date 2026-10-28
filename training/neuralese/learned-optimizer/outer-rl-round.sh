#!/bin/bash
# One outer meta-RL round of the joint learned optimizer with real task outcomes (run under the memory ledger by
# run-lfm3-lane.sh). usage: outer-rl-round.sh <operation dir with train-spec.json>
# PER_CONTEXT: at most this many kept component tokens per scope are scored beside each target's mixture (default 4). SCORE=skip (dry runs only) leaves the task outcomes out: the round then trains on the shaping reward alone.
# 1. proposals: per target its mixture and each scored kept token alone; each proposal and each target's incoming bundle becomes a release fragment.
# 2. natlang tune ROOT --score-proposals: each program seals and runs its proposals on its query roots.
# 3. collect: rewards = task outcome + shaping.  4. train: the exact expected-score update on those records.
set -euo pipefail
W=$(realpath "$1")
R=$W/outer-rl
PY=${PY:-/home/werg/natlang/.venv-neuralese/bin/python}
NATLANG=${NATLANG:-"/home/werg/.local/share/natlang-dev/node-v24.0.1-linux-arm64/bin/node /home/werg/natlang/ts-host/bin/natlang.mjs"}
mkdir -p "$R"
: > "$R/score.log"
"$PY" - "$W" "${PER_CONTEXT:-4}" <<'PY'
import json, sys
w = sys.argv[1]
train = json.load(open(f'{w}/train-spec.json'))
common = {k: train[k] for k in ('checkpoint', 'base', 'store', 'device', 'dtype', 'bank', 'targets', 'state') if k in train}
json.dump({**common, 'perContext': int(sys.argv[2]), 'out': f'{w}/outer-rl/proposals.jsonl', 'releases': f'{w}/outer-rl/releases'},
          open(f'{w}/outer-rl/proposals-spec.json', 'w'), indent=1)
json.dump({'proposals': f'{w}/outer-rl/proposals.jsonl', 'outcomes': f'{w}/outer-rl/outcomes.json',
           'out': f'{w}/outer-rl/records.jsonl'}, open(f'{w}/outer-rl/collect-spec.json', 'w'), indent=1)
json.dump({**train, 'steps': 0, 'policy_records': [f'{w}/outer-rl/records.jsonl']}, open(f'{w}/outer-rl/policy-spec.json', 'w'), indent=1)
PY
"$PY" -m natlang_neuralese.tune.learned proposals --spec "$R/proposals-spec.json" | tail -1 > "$R/proposals.json"
"$PY" -c 'import json, sys; [print(root, path) for root, path in json.load(open(sys.argv[1]))["result"]["requests"].items()]' \
  "$R/proposals.json" | while read -r root request; do
  [ "${SCORE:-}" = skip ] || $NATLANG tune "$root" --score-proposals "$request" >> "$R/score.log" 2>&1
done
"$PY" - "$R" <<'PY'
import glob, json, sys
r = sys.argv[1]
outcomes = {}
for path in sorted(glob.glob(f'{r}/releases/requests/*.outcomes.json')):
    outcomes.update(json.load(open(path)))
json.dump(outcomes, open(f'{r}/outcomes.json', 'w'))
PY
"$PY" -m natlang_neuralese.tune.learned collect --spec "$R/collect-spec.json" > "$R/collect.log" 2>&1
"$PY" -m natlang_neuralese.tune.learned train --spec "$R/policy-spec.json" > "$R/policy.log" 2>&1
