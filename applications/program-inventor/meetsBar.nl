---
description: Decide whether a built program meets one judged requirement.
model: reviewer
readout: decision
args:
  requirement: Requirement
  program: ProgramView
returns: boolean
---
program is a natlang program built for program.idea: its files (a root `.nl` function, `types.ts` and the root's
callable folder of `.nl` judgments and TypeScript helpers), its first input instance and its crisp metrics.

Read the files the requirement concerns, then return true when program meets requirement.statement as written, and
false when any part of the statement does not hold for it. Judge the program as it is, not the idea's description of
it.
