#!/bin/sh
# GRPO experiment boundary in two commands (plans/neuralese/DECISIONS.md "2026-11-13: grpo-v2").
#
#   grpo_boundary.sh pin COMMIT   pinned copy of COMMIT (scripts/pin_checkout.py), its held series, targeted tests in it;
#                                 writes $PIN/.boundary-ok when they pass. Runs as one ledger unit; this shell waits.
#                                 Touches no running unit.
#   grpo_boundary.sh switch       stop $OLD after its next step record (state saved), cancel any requeue of it, launch
#                                 $EXP-$RUN from the pin, append the switch to ops/decisions.log.
#   grpo_boundary.sh status       pins, units, last step records.
#
# Env: EXP=grpo-v2 RUN=run1 OLD=grpo-v1-run2 (unit to end) COMMIT12 (switch: which pin, default the only one)
#      WAIT_STEP=1 (switch: wait for OLD's next step record; 0 stops now) SLOTS=28 ACT_GB=6 KV_GB=6 BUDGET=100
#      EXTRA="--neuralese-variants ..." (later experiments: their one change).
set -e
EXP=${EXP:-grpo-v2}; RUN=${RUN:-run1}; OLD=${OLD:-grpo-v1-run2}
OPS=/home/werg/data/lfm3-rl-dgx-20261029/ops
W=$OPS/$EXP; OLD_WORK=$OPS/grpo-v1/run2
R=/home/werg/natlang; PYV=$R/.venv-neuralese/bin/python; LEDGER="/usr/bin/python3 $R/scripts/memory_ledger.py"
CKPT=/mnt/external/natlang-development-data/archive/data/lfm3-rl-dgx-20261029/fresh-init/checkpoint.pt
SHA=5ff07638bbedba7c8ec506f763ce5fb7511c7755ce8c045ef764bca00ddc1d6a
log() { echo "$(date '+%Y-%m-%d %H:%M') $*" >> $OPS/decisions.log; echo "$*"; }
wait_unit() { sleep 5; while systemctl --user is-active -q "$1"; do sleep 20; done; }  # ledger units are asynchronous
steps() { grep -c '"phase_s"' "$1/train.jsonl" 2>/dev/null || echo 0; }

case "$1" in
pin)
  C=$(git -C $R rev-parse "${2:?COMMIT}"); C12=$(echo $C | cut -c1-12); PIN=$W/$RUN/pin-$C12
  git -C $R fetch -q origin
  git -C $R merge-base --is-ancestor $C origin/main || { echo "$C12 is not on origin/main"; exit 2; }
  mkdir -p $W/$RUN
  UNIT=$EXP-$RUN-pin-$(date +%H%M%S)
  $LEDGER run --unit $UNIT --budget-gb 24 --priority medium --wait 3600 --workdir $R -- /bin/sh -c "
    set -e; exec >> $PIN.log 2>&1; /usr/bin/python3 $R/scripts/pin_checkout.py create $C $PIN
    cd $PIN; export NATLANG_REPO=$PIN PYTHONPATH=$PIN/training/neuralese:$PIN OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      NATLANG_HASH_CACHE=\$(mktemp -d) HF_HUB_OFFLINE=1
    $PYV -c 'import json; from natlang_neuralese.rl.held_eval import series; print(json.dumps(series()))' > $PIN.series.json
    $PYV -m pytest -q -x tests/neuralese/test_grpo_learner.py tests/neuralese/test_lfm3_fast_train.py tests/test_bonsai_queue.py
    touch $PIN/.boundary-ok" || true
  wait_unit $UNIT
  if [ -f $PIN/.boundary-ok ]; then
    echo "pin ok: $PIN series $(cat $PIN.series.json)"
  else
    tail -n 30 $PIN.log; echo "pin FAILED: $PIN (log $PIN.log)"; exit 1
  fi ;;
switch)
  if [ -n "$COMMIT12" ]; then PIN=$W/$RUN/pin-$COMMIT12; else PIN=$(ls -d $W/$RUN/pin-* | grep -v '\.' | tail -n 1); fi
  test -f $PIN/.boundary-ok || { echo "no tested pin ($PIN): run '$0 pin COMMIT' first"; exit 2; }
  if systemctl --user is-active -q $OLD; then
    if [ "${WAIT_STEP:-1}" = 1 ]; then
      n=$(steps $OLD_WORK); echo "waiting for $OLD step record $((n + 1))"
      while [ "$(steps $OLD_WORK)" -le "$n" ] && systemctl --user is-active -q $OLD; do sleep 60; done
      sleep 180   # the step's FP32 state save runs in a background thread
    fi
    systemctl --user stop $OLD
  fi
  $LEDGER cancel --unit $OLD >/dev/null 2>&1 || true
  for u in $(systemctl --user list-units --plain --no-legend "$OLD-*" | awk '{print $1}'); do systemctl --user stop "$u"; done
  E="/usr/bin/env NATLANG_REPO=$PIN PYTHONPATH=$PIN/training/neuralese:$PIN HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
  # grpo.py's defaults are the experiment (LR 1e-5, 24-48 groups, per-episode norm, graded reward, top-p 0.95, tail
  # guard, decisions as calls, errata excluded, 40 calls, explore 0.4); below: memory sizing measured on grpo-v1-run2,
  # Pop held evals, ext_codebase families back (e9a7360cc binds their repo snapshots).
  $LEDGER run --unit $EXP-$RUN --budget-gb ${BUDGET:-100} --fixed-claim --priority high --wait 3600 --detach --workdir $PIN -- \
    $E $PYV $PIN/scripts/neuralese_data/grpo.py train --checkpoint $CKPT --checkpoint-sha256 $SHA --work $W/$RUN \
    --slots ${SLOTS:-28} --activation-gb ${ACT_GB:-6} --actor-arg=--sglang-kv-gb --actor-arg=${KV_GB:-6} \
    --save-every 2 --state-every 1 --resume --state-dir $W/$RUN-state --checkpoint-dir /mnt/external/natlang-$EXP/$RUN/checkpoints \
    --held-remote pop --held-every-hours 4 --held-samples 2 --exclude-family-regex '(?!)' --unit-prefix $EXP-$RUN $EXTRA
  log "$EXP-$RUN launched from $PIN (series $(cat $PIN.series.json 2>/dev/null | cut -c1-200)) by grpo_boundary.sh switch; $OLD stopped after $(steps $OLD_WORK) step records${EXTRA:+; EXTRA=$EXTRA}" ;;
status)
  for f in $W/*/pin-*.series.json; do if [ -f "$f" ]; then echo "$f: $(cut -c1-120 $f)"; fi; done
  systemctl --user list-units --plain --no-legend 'grpo-*' | cut -c1-100
  echo "$OLD steps: $(steps $OLD_WORK)  $EXP-$RUN steps: $(steps $W/$RUN)" ;;
*) sed -n 2,14p "$0"; exit 2 ;;
esac
