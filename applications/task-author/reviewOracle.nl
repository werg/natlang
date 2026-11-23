---
description: Decide whether an authored task's oracle faithfully verifies the intended semantic outcome without loopholes.
args:
  task: AuthoredTask
returns: boolean
readout: decision
---
Inspect task.prompt, task.oracle, task.verifier, and task.files.
Return true if the oracle faithfully checks the task's core semantic goal without leaking answers or allowing trivial hardcoded shortcuts.
Return false if the oracle is ambiguous, leaks the exact target, or fails to verify the core requirement.
