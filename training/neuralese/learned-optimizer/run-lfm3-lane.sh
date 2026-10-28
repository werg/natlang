#!/bin/bash
# The LFM3 tuning and learned-optimizer lane, end to end on one trained LFM3 checkpoint (README.md beside this script).
#
#   run-lfm3-lane.sh <trained LFM3 full checkpoint .pt> [operation id]
#   run-lfm3-lane.sh --dry-run [operation id]      the same chain on a tiny fixture checkpoint (CPU, minutes)
#
# Stages, each its own memory-ledger unit lopt-<stage>-<operation> (resumable, medium priority) that starts only after
# the previous one recorded exit status 0 and printed its result line; statuses and logs live in the operation dir:
#   supply      privileged training references of every supplied program (training-partition roots only)
#   tune-<n>    per program: natlang tune --rounds with the default self teacher (privileged self-teaching on training
#               roots, oracle-reward RL rounds), development evaluation, tune result. A failed program is recorded in
#               tunes/failed.txt and the lane continues with the others.
#   specs       family split into optimizer corpus vs held evaluation programs; bank-init/corpus specs
#   bank-init, corpus
#   outer cycles (lane outer-next): train to the next bank publication; adopt the publication (adopt-bank) and refit
#               the labels against it (corpus into cycles/<k>/corpus); repeat until the recipe's steps
#   outer-rl    one outer meta-RL round with real task outcomes (natlang tune --score-proposals)
#   meta-examples  exact labels for the optimizer's own stages
# Rerunning the same operation id skips succeeded stages; an operation stays bound to its checkpoint SHA.
# Environment: LANE_ROOT (operations root), PROGRAMS (prepared supply), EXTRA_PROGRAMS (file: one further tune program
# directory per line, e.g. the operational tuning programs, whose earlier verified teacher rows are reused),
# PROTECTED_ROOT (checkout with the protected packets' cases, default this one), PY, NODE, TUNE_BUDGET_GB/STAGE_BUDGET_GB.
set -euo pipefail
HERE=$(dirname "$(realpath "$0")")
REPO=$(realpath "$HERE/../../..")
DRY=0
if [ "${1:-}" = --dry-run ]; then DRY=1; shift; fi
PY=${PY:-/home/werg/natlang/.venv-neuralese/bin/python}
NODE=${NODE:-/home/werg/.local/share/natlang-dev/node-v24.0.1-linux-arm64/bin/node}
if [ $DRY = 1 ]; then
  OP=${1:-dry-$(date -u +%Y%m%dT%H%M%SZ)}
  W=${LANE_ROOT:-/tmp/lopt-lfm3-lane-dry}
  TUNE_BUDGET_GB=${TUNE_BUDGET_GB:-6}; STAGE_BUDGET_GB=${STAGE_BUDGET_GB:-6}
else
  [ -n "${1:-}" ] || { echo "usage: $0 <trained LFM3 checkpoint .pt> [operation id] | --dry-run [operation id]" >&2; exit 2; }
  CK=$(realpath "$1")
  OP=${2:-lfm3-$(date -u +%Y%m%dT%H%M%SZ)}
  W=${LANE_ROOT:-/mnt/external/natlang-learned-optimizer-lfm3}
  PROGRAMS=${PROGRAMS:-/home/werg/data/program-supply-20261027/programs}
  TUNE_BUDGET_GB=${TUNE_BUDGET_GB:-48}; STAGE_BUDGET_GB=${STAGE_BUDGET_GB:-48}
fi
[[ $OP =~ ^[A-Za-z0-9-]+$ ]] || { echo "operation id must be letters, digits and dashes: $OP" >&2; exit 2; }
O=$W/operations/$OP
mkdir -p "$O/tunes" "$O/optimizer"
cd "$REPO"
ENV="/usr/bin/env PYTHONPATH=$REPO/training/neuralese:$REPO NATLANG_HASH_CACHE=${NATLANG_HASH_CACHE:-$O/hash-cache} NATLANG_CONFIG_HOME=$O/natlang-config"

# run_stage NAME BUDGET_GB COMMAND...: run COMMAND as ledger unit lopt-NAME-OP, wait for its own recorded exit status
# and return it. stage NAME BUDGET_GB RESULT COMMAND... additionally stops the run unless that status is 0 and the log
# ends with a '{"command": RESULT, ...}' line ('' for none).
run_stage() {
  local name=$1 budget=$2; shift 2
  local unit=lopt-$name-$OP status=$O/$name.status log=$O/$name.log
  if [ -f "$status" ] && [ "$(cat "$status")" = 0 ]; then echo "$name: already succeeded in $OP"; return 0; fi
  rm -f "$status"
  # The wrapper writes the status only when the command itself ends; a preempted unit is requeued and rewrites it.
  python3 scripts/memory_ledger.py run --unit "$unit" --budget-gb "$budget" --priority "${LANE_PRIORITY:-high}" --wait 86400 --resumable --workdir "$REPO" -- \
    /bin/bash -c "{ $*; } > $log 2>&1; rc=\$?; echo \$rc > $status.pending && mv $status.pending $status; exit \$rc" > /dev/null
  until [ -f "$status" ]; do
    sleep "${POLL_SECONDS:-20}"
    [ -f "$status" ] && break
    if ! systemctl --user is-active --quiet "$unit" && \
       ! python3 scripts/memory_ledger.py status | python3 -c 'import json,sys; u=sys.argv[1]+".service"; sys.exit(0 if any(w["unit"]==u for w in json.load(sys.stdin).get("waiting",[])) else 1)' "$unit" && \
       [ ! -f "$status" ]; then
      echo "$name: unit $unit left the ledger without a status; see $log" >&2; return 1
    fi
  done
  return "$(cat "$status")"
}
result_line() {  # result_line LOG RESULT: the log's last line is the RESULT command's JSON result
  tail -1 "$1" | python3 -c 'import json,sys; sys.exit(0 if json.loads(sys.stdin.read()).get("command")==sys.argv[1] else 1)' "$2" 2>/dev/null
}
stage() {
  local name=$1 budget=$2 result=$3 rc=0; shift 3
  run_stage "$name" "$budget" "$@" || rc=$?
  [ "$rc" = 0 ] || { echo "$name: exit status $rc; see $O/$name.log" >&2; exit 1; }
  if [ -n "$result" ] && ! result_line "$O/$name.log" "$result"; then
    echo "$name: succeeded without its $result result line; see $O/$name.log" >&2; exit 1
  fi
  echo "$name: succeeded"
}

if [ $DRY = 1 ]; then
  stage fixture "$STAGE_BUDGET_GB" fixture $ENV $PY "$HERE/dry_run_fixture.py" build --out "$O/fixture"
  CK=$O/fixture/model/native.pt
  PROGRAMS=$O/fixture/programs
fi
SHA=$($ENV "$PY" -c 'from natlang_neuralese.common.checkpoint_digest import verified_checkpoint_digest; import sys; print(verified_checkpoint_digest(sys.argv[1]))' "$CK")
if [ -f "$O/operation.json" ]; then
  python3 - "$O/operation.json" "$CK" "$SHA" <<'PY'
import json, sys
op = json.load(open(sys.argv[1]))
assert (op['checkpoint'], op['checkpoint_sha256']) == (sys.argv[2], sys.argv[3]), \
    f"operation {sys.argv[1]} is bound to {op['checkpoint']} ({op['checkpoint_sha256']}); use a new operation id"
PY
else
  python3 -c 'import json,sys,time; json.dump({"operation": sys.argv[2], "checkpoint": sys.argv[3], "checkpoint_sha256": sys.argv[4], "programs": sys.argv[5], "dry_run": sys.argv[6] == "1", "started": time.time()}, open(sys.argv[1], "w"), indent=1)' \
    "$O/operation.json" "$OP" "$CK" "$SHA" "$PROGRAMS" "$DRY"
fi

# The tuner's configuration: the trained checkpoint is the student (native backend) and its own privileged teacher.
if [ $DRY = 0 ] && [ ! -f "$O/natlang-config/config.json" ]; then
  mkdir -p "$O/natlang-config"
  python3 - "$O/natlang-config/config.json" "$CK" "$SHA" "$PY" "$REPO" <<'PY'
import json, sys
path, checkpoint, sha, python, repo = sys.argv[1:]
json.dump({'defaultProfile': 'lfm3-lane', 'profiles': {'lfm3-lane': {'native': {
    'checkpoint': checkpoint, 'sha256': sha, 'python': python, 'pythonPath': [f'{repo}/training/neuralese', repo],
    'device': 'cuda', 'dtype': 'bfloat16'}}},
    # Reply/action targets only (LFM3 recipe: prompts are zero-weight context); passes over the admitted rows instead
    # of the trainer's one-step default; RL rounds continue with one pass of sft/rl/rft each.
    # Invocation limits as in the operational tuning lane (PROGRAM_TUNING_RUNS_20261026.md).
    'tune': {'teacher': 'self', 'backend': 'native', 'softStates': True,
             'execution': {'maxTurns': 12, 'maxTokens': 4096, 'maxSeconds': 120, 'maxFailureRepairs': 3},
             'recipe': {'epochs': 3, 'batch_rows': 8, 'prompt_weight': 0},
             'rounds': {'schedule': ['sft', 'rl', 'rl', 'rl'], 'samples': 4, 'temperature': 0.8, 'patience': 2,
                        'rlRecipe': {'epochs': 1}}}},
    open(path, 'w'), indent=1)
PY
fi

# Privileged facts through the shared JOLT+ mechanism; custody against every registered evaluation-only packet of
# this checkout's registry, read under PROTECTED_ROOT (the checkout holding runs/<packet>/cases.jsonl; fails closed).
stage supply "$STAGE_BUDGET_GB" privileged $ENV $PY -m natlang_neuralese.tune.supply privileged --programs "$PROGRAMS" \
  --registry "$REPO/training/protected-worlds.json" --root "${PROTECTED_ROOT:-$REPO}"

# Program tunes, one ledger unit per program (CoreWeave sandboxes can run disjoint program shards of this loop).
mapfile -t TUNE_PROGRAMS < <(python3 -c 'import json,sys; [print(sys.argv[1] + "/" + item["program"]) for item in json.load(open(sys.argv[1] + "/programs-prepared.json"))["programs"]]' "$PROGRAMS")
if [ -n "${EXTRA_PROGRAMS:-}" ]; then mapfile -t -O "${#TUNE_PROGRAMS[@]}" TUNE_PROGRAMS < <(grep -v '^\s*$' "$EXTRA_PROGRAMS"); fi
: > "$O/tunes/failed.txt"
NATLANG="$NODE $REPO/ts-host/bin/natlang.mjs"
for index in "${!TUNE_PROGRAMS[@]}"; do
  program=${TUNE_PROGRAMS[$index]}; name=$(basename "$program")
  if [ $DRY = 1 ]; then
    command="$ENV $PY $HERE/dry_run_fixture.py tune --program $program --checkpoint $CK --out $O/tunes/$name.json"
  else
    command="$ENV $NATLANG tune $program --rounds && $ENV $NATLANG tune $program --evaluate development && \
$ENV $PY -m natlang_neuralese.tune.learned.lane tune-result --program $program --out $O/tunes/$name.json"
  fi
  rc=0; run_stage "tune-$index" "$TUNE_BUDGET_GB" "$command" || rc=$?
  if [ "$rc" != 0 ] || ! result_line "$O/tune-$index.log" tune-result; then
    echo "$name tune-$index exit $rc" >> "$O/tunes/failed.txt"; echo "tune $name: failed (exit $rc); see $O/tune-$index.log" >&2
  else
    echo "tune $name: succeeded"
  fi
done

EXTRA=''
[ $DRY = 1 ] && EXTRA=$(python3 -c 'import json; print(json.dumps({
 "*": {"device": "cpu", "dtype": "float32"},
 "bank-init": {"rank": 4, "per_source_rank": 2, "component_rank": 2, "component_steps": 2},
 "corpus": {"fit": {"steps": 2, "refine_steps": 1}, "subsets": 2, "max_support": 4, "incremental_max": 1, "recipe": {"batch_rows": 4}},
 "train": {"steps": 3, "recipe": {"writer": {"kind": "free-lora", "rank": 2, "alpha": 2.0}, "batch_targets": 1, "publish_every": 2}},
 "meta": {"steps": 2, "refine_steps": 1, "fit": {"steps": 2, "refine_steps": 1},
          "learned": {"scopes": {"tree": 3, "folder": 3, "lambda": 3}, "steps": 2, "patience": 2}}}))')
stage specs "$STAGE_BUDGET_GB" specs $ENV $PY -m natlang_neuralese.tune.learned.lane specs --op "$O" --checkpoint "$CK" \
  --templates "$HERE" $([ $DRY = 1 ] && echo "--held-fraction 0") ${EXTRA:+--extra "'$EXTRA'"}
stage bank-init "$STAGE_BUDGET_GB" bank-init $ENV $PY -m natlang_neuralese.tune.learned bank-init --spec "$O/bank-init-spec.json"
stage corpus "$STAGE_BUDGET_GB" corpus $ENV $PY -m natlang_neuralese.tune.learned corpus --spec "$O/corpus-spec.json"

# Outer training cycles: train to each bank publication, adopt it, refit the labels against it, continue. They run
# again after the meta-RL round: its policy update can publish a bank, which is adopted and refit before meta-examples.
outer_cycles() {
  local NEXT
  while :; do
    NEXT=$($ENV $PY -m natlang_neuralese.tune.learned.lane outer-next --op "$O" | tail -1)
    field() { python3 -c 'import json,sys; print(json.loads(sys.argv[1]).get(sys.argv[2], ""))' "$NEXT" "$1"; }
    case $(field action) in
      train) stage "outer-train-c$(field cycle)-s$(field step)" "$STAGE_BUDGET_GB" train \
               $ENV $PY -m natlang_neuralese.tune.learned train --spec "$(field spec)" ;;
      adopt) stage "adopt-c$(field cycle)" "$STAGE_BUDGET_GB" adopt-bank $ENV $PY -m natlang_neuralese.tune.learned adopt-bank --spec "$(field adopt)"
             stage "corpus-c$(field cycle)" "$STAGE_BUDGET_GB" corpus $ENV $PY -m natlang_neuralese.tune.learned corpus --spec "$(field corpus)"
             touch "$(field complete)" ;;
      done)  echo "outer training: done at step $(field step) after $(field cycle) adopted publication(s)"; return 0 ;;
      *)     echo "outer-next gave no action: $NEXT" >&2; exit 1 ;;
    esac
  done
}
outer_cycles

# Outer meta-RL with real task outcomes (dry runs: shaping reward only), then the optimizer's own stage labels.
stage outer-rl "$STAGE_BUDGET_GB" '' $ENV PY=$PY "NATLANG='$NATLANG'" PER_CONTEXT=$([ $DRY = 1 ] && echo 2 || echo 4) \
  $([ $DRY = 1 ] && echo SCORE=skip) $HERE/outer-rl-round.sh "$O"
result_line "$O/outer-rl/policy.log" train || { echo "outer-rl: the policy update did not finish; see $O/outer-rl/policy.log" >&2; exit 1; }
outer_cycles
stage meta-examples "$STAGE_BUDGET_GB" meta-examples $ENV $PY -m natlang_neuralese.tune.learned meta-examples --spec "$O/meta-spec.json"
if [ $DRY = 0 ]; then
  CONTROLLER=$O/controller
  if [ ! -d "$CONTROLLER" ]; then cp -a "$REPO/applications/learned-optimizer" "$CONTROLLER"; fi
  stage controller-split "$STAGE_BUDGET_GB" controller-split $ENV $NODE "$HERE/controller-split.mjs" "$CONTROLLER" "$O/meta/tune.examples.jsonl"
  stage controller-supply "$STAGE_BUDGET_GB" controller-supply $ENV $PY "$HERE/controller-supply.py" --controller "$CONTROLLER" \
    --examples "$O/meta/tune.examples.jsonl" --root "${PROTECTED_ROOT:-$REPO}"
  stage controller-tune "$TUNE_BUDGET_GB" '' $ENV NATLANG_LOPT_REPOSITORY=$REPO NATLANG_LOPT_PYTHON=$PY \
    $NATLANG tune "$CONTROLLER" --examples "$O/meta/tune.examples.jsonl" --rounds
  $PY - "$O" "$CONTROLLER" <<'PY'
import json, pathlib, sys
op, root = map(pathlib.Path, sys.argv[1:])
active = json.loads((root / '.natlang/tune/active.json').read_text())
learned = json.loads((op / 'meta-spec.json').read_text())['learned']
learned['controller'] = {'root': str(root), 'release': active['release']}
path = op / 'learned-controller.json'
if path.exists() and json.loads(path.read_text()) != learned:
    raise SystemExit('controller binding changed: use a new operation')
path.write_text(json.dumps(learned, indent=2) + '\n')
PY
  mapfile -t HELD < <($PY -c 'import json,sys; print("\n".join(json.load(open(sys.argv[1]))["held"]))' "$O/optimizer-split.json")
  for index in "${!HELD[@]}"; do
    name=${HELD[$index]}
    program=$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]))["root"])' "$O/tunes/$name.json")
    baseline=$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]))["best"]["release"])' "$O/tunes/$name.json")
    stage "held-learned-$index" "$TUNE_BUDGET_GB" '' $ENV NATLANG_LOPT_REPOSITORY=$REPO NATLANG_LOPT_PYTHON=$PY \
      $NATLANG tune "$program" --learned "$O/learned-controller.json"
    stage "held-development-$index" "$TUNE_BUDGET_GB" '' $ENV $NATLANG tune "$program" --evaluate development
    stage "held-confirmation-$index" "$TUNE_BUDGET_GB" '' $ENV $NATLANG tune "$program" --evaluate confirmation
    stage "held-baseline-development-$index" "$TUNE_BUDGET_GB" '' $ENV $NATLANG tune "$program" --evaluate development --release "$baseline"
    stage "held-baseline-confirmation-$index" "$TUNE_BUDGET_GB" '' $ENV $NATLANG tune "$program" --evaluate confirmation --release "$baseline"
  done
fi
$ENV $PY -m natlang_neuralese.tune.learned.lane summary --op "$O" | tee "$O/summary.json"
