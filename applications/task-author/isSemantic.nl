---
description: Decide whether an authored task genuinely requires semantic reasoning rather than keyword rules.
args:
  task: AuthoredTask
returns: boolean
readout: decision
---
Inspect task.prompt and task.files.
Return true if solving the task genuinely requires understanding meaning, nuance, intent, or semantic structure.
Return false if a brittle keyword matcher, regex, or purely mechanical arithmetic/sorting script could solve it.
