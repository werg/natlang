---
args:
  state: State
  event: UiEvent
returns: Decision
---
Design matched seeded inventory experiments. hypothesis stores text. run uses target policy cautious or generous, amount seed (nonnegative integer), count steps (positive integer). The natlang caller compares policies by calling run repeatedly with matched seed and horizon. Exact simulation samples demand 0..9 per step; cautious stocks 4, generous stocks 8; reward is 3*units sold - stock - 2*unserved demand. Explain tradeoffs only from recorded trials. No arbitrary episode limit is imposed.

For an explicit control event, event.kind is the requested action and event.value
is a JSON record of its fields. Preserve those fields and do not substitute a
different action. For a command event, interpret the user's request using the
current state and the documented actions. Return the next applicable Decision. The caller owns sequencing and completion.
Keep edits and semantic judgments within the user's request. Host effects and
results belong to apply; never fabricate an execution result here.
