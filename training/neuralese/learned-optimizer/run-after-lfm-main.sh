#!/bin/bash
# HOLD (owner, 2026-10-27): run only after LFM main training is done.
# usage: run-after-lfm-main.sh <final LFM full checkpoint .pt>
# Rebinds the prepared specs to the final checkpoint, then queues, one after another in the memory ledger:
# bank-init -> corpus (tuner fits on the bank basis + joint targets) -> outer training -> outer meta-RL with real task
# outcomes -> meta-examples (exact labels
# for the optimizer's own stages). See README.md beside this script. Supplied programs need case export, teacher
# collection and import first (program-tuning lane); add their imported tune plans to corpus-spec.json "programs".
set -euo pipefail
W=/mnt/external/natlang-learned-optimizer-20261027
CK=$(realpath "$1")
# The checked-in specs (lfm-*-spec.json beside this script) are the templates; existing operation specs are kept.
mkdir -p $W
for name in bank-init-spec.json corpus-spec.json train-spec.json meta-spec.json; do
  [ -f $W/$name ] || cp "$(dirname "$(realpath "$0")")/lfm-$name" $W/$name
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
cd /home/werg/natlang
ENV="/usr/bin/env PYTHONPATH=/home/werg/natlang/training/neuralese:/home/werg/natlang NATLANG_HASH_CACHE=$W/hash-cache"
PY=/home/werg/natlang/.venv-neuralese/bin/python
python3 scripts/memory_ledger.py run --unit lopt-bank-init-final --budget-gb 20 --wait 86400 --resumable --workdir /home/werg/natlang -- \
  $ENV /bin/bash -c "$PY -m natlang_neuralese.tune.learned bank-init --spec $W/bank-init-spec.json > $W/bank-init.log 2>&1"
# Ledger units run asynchronously: the corpus starts only once the bank exists.
until [ -f $W/bank.json ]; do sleep 60; done
python3 scripts/memory_ledger.py run --unit lopt-corpus-final --budget-gb 20 --wait 86400 --resumable --workdir /home/werg/natlang -- \
  $ENV /bin/bash -c "$PY -m natlang_neuralese.tune.learned corpus --spec $W/corpus-spec.json >> $W/corpus.log 2>&1"
# Outer training once the corpus has finished labeling (its log ends with the command's JSON result line).
until grep -q '"command": "corpus"' $W/corpus.log 2>/dev/null; do sleep 120; done
mkdir -p $W/optimizer
python3 scripts/memory_ledger.py run --unit lopt-outer-train-final --budget-gb 20 --wait 86400 --resumable --workdir /home/werg/natlang -- \
  $ENV /bin/bash -c "$PY -m natlang_neuralese.tune.learned train --spec $W/train-spec.json >> $W/train.log 2>&1"
# Outer meta-RL with real task outcomes, after supervised outer training: sample proposals per target, seal and run
# each on its program's query roots (natlang tune --score-proposals), then the policy update on those rewards.
until grep -q '"command": "train"' $W/train.log 2>/dev/null; do sleep 120; done
python3 scripts/memory_ledger.py run --unit lopt-outer-rl-final --budget-gb 24 --wait 86400 --resumable --workdir /home/werg/natlang -- \
  $ENV /bin/bash -c "/home/werg/natlang/training/neuralese/learned-optimizer/outer-rl-round.sh $W >> $W/outer-rl.log 2>&1"
# Meta-learner examples (exact hindsight/counterfactual labels for its own stages) after the outer meta-RL round.
until grep -q '"command": "train"' $W/outer-rl/policy.log 2>/dev/null; do sleep 120; done
python3 scripts/memory_ledger.py run --unit lopt-meta-examples-final --budget-gb 20 --wait 86400 --resumable --workdir /home/werg/natlang -- \
  $ENV /bin/bash -c "$PY -m natlang_neuralese.tune.learned meta-examples --spec $W/meta-spec.json >> $W/meta.log 2>&1"
# Then: tune the optimizer program itself with $W/meta/tune.examples.jsonl (plans/neuralese/JOINT_LEARNED_OPTIMIZER_STATUS.md).
