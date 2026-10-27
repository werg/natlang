---
description: One revision of the complete adaptation. Interpret the evidence, choose what to revise and how, ask the optimizer service for a complete candidate bundle, and keep it only when its measured support loss is lower.
args:
  state: AdaptationState
  context: AdaptationContext
uses: [adapt/interpretEvidence, adapt/chooseRevision, adapt/chooseOperator]
returns: AdaptationState
---
Make one revision of the program's whole adaptation: every instruction state, skill state and LoRA mixture is
revised together. Work in eval and return the next state.

1. analysis = interpretEvidence(context.view, state.history).
2. plan = chooseRevision(analysis.focus, context.view, state.history).
3. operator = chooseOperator(state.history, plan, state.remainingSteps).
4. parent = state.current ? state.current.id : null. step = context.maxSteps - state.remainingSteps.
5. Run the operator:
   - "learned-proposal": candidate = await optimizer.propose({ parent, sample: step > 0, seed: step,
     focus: analysis.focus, digests: analysis.digests }).
   - "retrieval": candidate = await optimizer.retrieve({ parent }).
   - "local-refinement": candidate = await optimizer.refine({ parent, steps: 10 }).
   - "keep": candidate = null.
6. best = state.current ? state.current.supportLoss : state.incomingLoss. accepted = candidate is not null and
   candidate.supportLoss < best.
7. Return { ...state, current: accepted ? candidate : state.current, remainingSteps: state.remainingSteps - 1,
   history: state.history plus { step, operator, bundle: candidate ? candidate.id : null,
   supportLoss: candidate ? candidate.supportLoss : null, accepted, reason: plan.reason } }.
   A rejected candidate stays in the history: later steps learn from it.
