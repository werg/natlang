---
description: One training cycle. Schedule the work from the metrics, run it through the trainer services in order, and record what ran.
args:
  state: TrainingState
uses: [train/scheduleTraining]
returns: TrainingState
---
1. schedule = scheduleTraining(state.metrics).
2. Run each entry of schedule.work in order:
   - "refit-labels": await trainer.refitLabels().
   - "supervised": await trainer.supervised({ steps: schedule.supervisedSteps }).
   - "collect-proposals": await trainer.collectProposals({ perContext: 4 }).
   - "policy": await trainer.policy().
   - "publish-bank": await trainer.publishBank().
   - "ladder": await trainer.ladder().
   - "done": run nothing.
3. metrics = await trainer.metrics().
4. Return { cycle: state.cycle + 1, remainingCycles: state.remainingCycles - 1, metrics,
   log: state.log plus { cycle: state.cycle, work: schedule.work, reason: schedule.reason } }.
