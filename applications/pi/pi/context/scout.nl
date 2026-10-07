---
description: Pick the files a coding task most likely needs.
args:
  task: string
  files: string[]
returns: string[]
model: small
---
files lists the project's files. Pick up to 8 that a developer would most likely need to read or change for task,
most likely first: the code the task names or describes, its tests, and the configuration that decides how it
builds or runs. Answer with paths from files only.
