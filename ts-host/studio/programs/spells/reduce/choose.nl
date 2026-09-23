---
args:
  state: State
  event: UiEvent
returns: Decision
---
Translate spells into exact actions. cast uses words in text, element in target (fire, ice, wind, light), power amount (1 to 10). Cost is twice power; host checks mana and resolves damage against the sentinel ward. Use creative wording to select element/power, but never invent mechanics. rest refills mana and resets sentinel. The typed action is the proposed compilation; exact physics is host code.

For an explicit control event, event.kind is the requested action and event.value
is a JSON record of its fields. Preserve those fields and do not substitute a
different action. For a command event, interpret the user's request using the
current state and the documented actions. Return the next applicable Decision. The caller owns sequencing and completion.
Keep edits and semantic judgments within the user's request. Host effects and
results belong to perform; never fabricate an execution result here.
