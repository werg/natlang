---
kind: directory-reducer
args:
  state: CounterexampleState
  goal: string
returns: CounterexampleState
---
Perform one counterexample-guided repair experiment. The suite's expected outcomes belong to an independent author-provided oracle. Never supply or alter gold yourself.

Read folder.snapshot() and ask counterexamples.evidence for its training failures. Call suggestCounterexamples with goal and that evidence; it returns {inputs, reason}, inputs exposing a missing behavior. Ask counterexamples.admit to check suggestion.inputs against the independent oracle. It returns the new suite identity, admitted count and remaining check allowance; validation and test cases remain pinned. If none were admitted, the source is retained and there is no repair.

If examples were admitted, call counterexamples.repair(folder.snapshot()) to run the ordinary authored improvement program on the new suite. Install its actual returned source with await folder.select(result.folder), and record its disposition and measured quality.

End every round, with or without a repair, by calling counterexamples.stop(). The host measured the round (the admitted count, the remaining allowance and the repair) and the shouldStop policy decides from those facts, returning {stop, reason}. Set done to stop and reason to its reason. Increment state.round exactly once. Return the typed next state with the actual suite identity, remaining checks, quality, done and reason. Never invent additional capacity or evaluate a target yourself.
