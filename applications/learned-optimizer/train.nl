---
description: The outer training operation of the learned optimizer. Each cycle reads the measured training metrics, schedules the work that helps most now, and runs it through the exact trainer services.
args:
  cycles: number
returns: TrainingState
---
Train the learned optimizer for at most cycles cycles with the stage in your folder. The trainer service computes
gradients, applies updates, refits labels, publishes the bank and measures the ceiling ladder; you schedule the work.
Work in eval and return the final state.

1. metrics = await trainer.metrics().
2. state = { cycle: 0, remainingCycles: cycles, metrics, log: [] }.
3. final = await trainStep.iterateOn(state).withMeasure(s => s.remainingCycles).until(s => s.remainingCycles === 0 ||
   (s.log.length > 0 && s.log[s.log.length - 1].work.includes("done"))). When the loop ends with an error whose name is
   "IterationLimitError", final = the error's lastState.
4. Return final.
