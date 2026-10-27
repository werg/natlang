#!/bin/bash
# HOLD (owner, 2026-10-27): run only after LFM main training is done.
# usage: run-after-lfm-main.sh <final LFM full checkpoint .pt> [operation id]
# Rebinds the prepared specs to the final checkpoint, then runs, one after another in the memory ledger:
# bank-init -> corpus (tuner fits on the bank basis + joint targets) -> outer training -> outer meta-RL with real task
# outcomes -> meta-examples (exact labels
# for the optimizer's own stages). See README.md beside this script. Supplied programs need case export, teacher
# collection and import first (program-tuning lane); add their imported tune plans to corpus-spec.json "programs".
#
# Each stage is its own ledger unit named after the operation, logs into the operation directory and records its
# actual exit status there; the next stage starts only after that status is 0 and the stage printed its own result
# line. A failed stage, or a unit that leaves the ledger without a status, stops the run. Rerunning with the same
# operation id skips stages already recorded as succeeded (for the same checkpoint) and retries the rest; a new id
# starts a fresh namespace. Outer training that publishes a bank marks its labels stale: the run then stops with
# needs-refit.json instead of continuing on labels fitted against the previous bank snapshot.
set -euo pipefail
W=/mnt/external/natlang-learned-optimizer-20261027
CK=$(realpath "$1")
OP=${2:-lopt-$(date -u +%Y%m%dT%H%M%SZ)}
[[ $OP =~ ^[A-Za-z0-9-]+$ ]] || { echo "operation id must be letters, digits and dashes: $OP" >&2; exit 2; }
O=$W/operations/$OP
HERE=$(dirname "$(realpath "$0")")
cd /home/werg/natlang
SHA=$(sha256sum "$CK" | cut -c1-64)
mkdir -p "$O"
if [ -f "$O/operation.json" ]; then
  python3 - "$O/operation.json" "$CK" "$SHA" <<'PY'
import json, sys
op = json.load(open(sys.argv[1]))
assert (op['checkpoint'], op['checkpoint_sha256']) == (sys.argv[2], sys.argv[3]), \
    f"operation {sys.argv[1]} is bound to {op['checkpoint']} ({op['checkpoint_sha256']}); use a new operation id"
PY
else
  printf '{"operation": "%s", "checkpoint": "%s", "checkpoint_sha256": "%s", "started": %s}\n' \
    "$OP" "$CK" "$SHA" "$(date +%s)" > "$O/operation.json"
fi
# The checked-in specs (lfm-*-spec.json beside this script) are the templates; existing operation specs are kept.
for name in bank-init-spec.json corpus-spec.json train-spec.json meta-spec.json; do
  [ -f $W/$name ] || cp "$HERE/lfm-$name" $W/$name
done
python3 - "$W" "$CK" <<'PY'
import json, sys
w, ck = sys.argv[1:]
for name in ('bank-init-spec.json', 'corpus-spec.json', 'train-spec.json'):
    spec = json.load(open(f'{w}/{name}'))
    spec['checkpoint'] = ck
    if name == 'corpus-spec.json':
        spec['rebase'] = True
    json.dump(spec, open(f'{w}/{name}', 'w'), indent=1)
PY
ENV="/usr/bin/env PYTHONPATH=/home/werg/natlang/training/neuralese:/home/werg/natlang NATLANG_HASH_CACHE=$W/hash-cache"
PY=/home/werg/natlang/.venv-neuralese/bin/python

# run_stage NAME BUDGET_GB COMMAND...: run COMMAND as ledger unit lopt-NAME-OP, wait for its own recorded exit status
# and return it. stage NAME BUDGET_GB RESULT_COMMAND COMMAND... additionally stops the run unless that status is 0 and
# the log ends with the learned CLI's '{"command": RESULT_COMMAND, ...}' line ('' for none).
run_stage() {
  local name=$1 budget=$2; shift 2
  local unit=lopt-$name-$OP status=$O/$name.status log=$O/$name.log
  if [ -f "$status" ] && [ "$(cat "$status")" = 0 ]; then echo "$name: already succeeded in $OP"; return 0; fi
  rm -f "$status"
  # The wrapper writes the status only when the command itself ends; a preempted unit is requeued and rewrites it.
  python3 scripts/memory_ledger.py run --unit "$unit" --budget-gb "$budget" --wait 86400 --resumable --workdir /home/werg/natlang -- \
    /bin/bash -c "$* > $log 2>&1; rc=\$?; echo \$rc > $status.pending && mv $status.pending $status; exit \$rc"
  until [ -f "$status" ]; do
    sleep 60
    [ -f "$status" ] && break
    if ! systemctl --user is-active --quiet "$unit" && \
       ! python3 scripts/memory_ledger.py status | python3 -c 'import json,sys; u=sys.argv[1]+".service"; sys.exit(0 if any(w["unit"]==u for w in json.load(sys.stdin).get("waiting",[])) else 1)' "$unit" && \
       [ ! -f "$status" ]; then
      echo "$name: unit $unit left the ledger without a status; see $log" >&2; exit 1
    fi
  done
  return "$(cat "$status")"
}
stage() {
  local name=$1 budget=$2 result=$3 rc=0; shift 3
  run_stage "$name" "$budget" "$@" || rc=$?
  [ "$rc" = 0 ] || { echo "$name: exit status $rc; see $O/$name.log" >&2; exit 1; }
  local log=$O/$name.log
  if [ -n "$result" ] && ! tail -1 "$log" | python3 -c 'import json,sys; sys.exit(0 if json.loads(sys.stdin.read()).get("command")==sys.argv[1] else 1)' "$result"; then
    echo "$name: succeeded without its $result result line; see $log" >&2; exit 1
  fi
  echo "$name: succeeded"
}

stage bank-init 20 bank-init $ENV $PY -m natlang_neuralese.tune.learned bank-init --spec $W/bank-init-spec.json
stage corpus 20 corpus $ENV $PY -m natlang_neuralese.tune.learned corpus --spec $W/corpus-spec.json
mkdir -p $W/optimizer
rc=0; run_stage outer-train 20 $ENV $PY -m natlang_neuralese.tune.learned train --spec $W/train-spec.json || rc=$?
# A bank publication marks the coefficient labels stale (the trainer refuses the next update and saves that state):
# they must be refitted against the published snapshot before any further outer update or meta-RL.
STATE=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["state"])' $W/train-spec.json)
if [ -f "$STATE" ] && python3 -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("labels_stale") else 1)' "$STATE"; then
  python3 -c 'import json,sys,time; s=json.load(open(sys.argv[1])); json.dump({"at": time.time(), "operation": sys.argv[3], "exit_status": int(sys.argv[4]), "step": s.get("step"), "published_banks": s.get("published_banks", []), "reason": "outer training published a bank; refit coefficient labels against it before further updates"}, open(sys.argv[2], "w"), indent=1)' \
    "$STATE" "$O/needs-refit.json" "$OP" "$rc"
  echo "outer-train published a bank: labels need a refit before further training ($O/needs-refit.json)" >&2; exit 3
fi
[ "$rc" = 0 ] || { echo "outer-train: exit status $rc; see $O/outer-train.log" >&2; exit 1; }
tail -1 "$O/outer-train.log" | python3 -c 'import json,sys; sys.exit(0 if json.loads(sys.stdin.read()).get("command")=="train" else 1)' \
  || { echo "outer-train: succeeded without its train result line; see $O/outer-train.log" >&2; exit 1; }
# Outer meta-RL with real task outcomes, after supervised outer training: sample proposals per target, seal and run
# each on its program's query roots (natlang tune --score-proposals), then the policy update on those rewards.
stage outer-rl 24 '' $ENV $HERE/outer-rl-round.sh $W
tail -1 "$W/outer-rl/policy.log" | python3 -c 'import json,sys; sys.exit(0 if json.loads(sys.stdin.read()).get("command")=="train" else 1)' \
  || { echo "outer-rl: policy update did not finish; see $W/outer-rl/policy.log" >&2; exit 1; }
# Meta-learner examples (exact hindsight/counterfactual labels for its own stages) after the outer meta-RL round.
stage meta-examples 20 meta-examples $ENV $PY -m natlang_neuralese.tune.learned meta-examples --spec $W/meta-spec.json
# Then: tune the optimizer program itself with $W/meta/tune.examples.jsonl (plans/neuralese/JOINT_LEARNED_OPTIMIZER_STATUS.md).
