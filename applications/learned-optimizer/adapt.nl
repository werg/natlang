---
description: One adaptation episode of the joint learned optimizer. Read the program's permitted evidence, then revise its native instruction states, skill states and LoRA mixtures together, step by step, keeping the best measured bundle.
args:
  maxSteps: number
returns: AdaptationState
---
Adapt this program with the stages in your folder. The optimizer service holds the bundles, writes them and scores
them exactly; you decide what to revise and which operator to use. Work in eval and return the final state.

1. Evidence. seen = await optimizer.evidence(). context = { program: seen.view.id, view: seen.view, scopes: seen.scopes,
   maxSteps }.
2. Start. state = { current: null, incomingLoss: seen.incomingLoss, remainingSteps: maxSteps, history: [],
   stopped: false, stopReason: "" }. current null means the incoming bundle.
3. Revise. final = await adaptStep.iterateOn(state, context).withMeasure(s => s.remainingSteps).until(shouldStop).
   Each step lowers remainingSteps by one. When the loop ends with an error whose name is "IterationLimitError",
   final = the error's lastState: the step budget is spent and its best bundle is the result.
4. Return final. Its current bundle is the best one measured on the support evidence, or null when no revision beat
   the incoming bundle.
