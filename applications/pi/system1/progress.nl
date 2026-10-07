---
description: Judge whether a coding agent's recent actions make progress.
readout: decision
args:
  task: string
  recent: string[]
returns: Progress
---
recent lists a coding agent's latest actions on task, oldest first, each with what it observed. progressing: the
actions build on each other or narrow down the problem. repeating: it runs the same commands or makes the same edit
again with the same result. stuck: it keeps failing in different ways without learning anything new.
