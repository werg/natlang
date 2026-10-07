---
description: Judge whether a coding agent's final answer completes its task.
readout: decision
args:
  task: string
  answer: string
  evidence: string
returns: Completion
model: small
---
A coding agent says it is finished with task and answers with answer. evidence lists what it did and what its last
checks showed. done: the task's requests are all carried out and nothing in evidence contradicts it (tests it ran
pass, files it was asked to change are changed). unfinished: some request is not addressed, a check it ran still
fails, or it stopped to ask about something it could have found out itself.
