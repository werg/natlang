---
args:
  state: State
  event: UiEvent
returns: Decision
---
Use the real offline package registry. browse reads the companion catalog. resolve chooses package target and semver range text, engine typescript-host, and stores an exact content-addressed lock. install installs the current lock under target name through atomic pointer creation; installed names are immutable. Never invent compatibility or package contents. The initial registry ships small executable example bundles.

For an explicit control event, event.kind is the requested action and event.value
is a JSON record of its fields. Preserve those fields and do not substitute a
different action. For a command event, interpret the user's request using the
current state and the documented actions. Return the next applicable Decision. The caller owns sequencing and completion.
Keep edits and semantic judgments within the user's request. Host effects and
results belong to apply; never fabricate an execution result here.
