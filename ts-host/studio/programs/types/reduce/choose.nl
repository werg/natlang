---
args:
  state: State
  event: UiEvent
returns: Decision
---
Infer and review semantic types from the source. save stores source text and invalidates previous findings. infer adds target function ID, proposed TS-style signature text and evidence secondary. diagnostic adds location target and concrete mismatch text. clear discards findings. Do not claim exact static verification from a semantic judgment. Account for call sites and distinguish unknown host types using escape hatches.

For an explicit control event, event.kind is the requested action and event.value
is a JSON record of its fields. Preserve those fields and do not substitute a
different action. For a command event, interpret the user's request using the
current state and the documented actions. Return the next applicable Decision. The caller owns sequencing and completion.
Keep edits and semantic judgments within the user's request. Host effects and
results belong to perform; never fabricate an execution result here.
