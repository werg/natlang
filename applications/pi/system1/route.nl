---
description: Decide whether a coding agent's next step is routine enough for a small model.
readout: decision
args:
  task: string
  recent: string[]
returns: Route
---
recent lists a coding agent's latest actions on task, oldest first, each with what it observed. routine: the next
step follows directly from the last result: rerunning a command after a fix, reading a file an error names, making
an edit the agent already described, or answering when the last check passed. hard: it needs a new diagnosis, a
design choice, or a change whose correctness is not obvious from what is shown.
