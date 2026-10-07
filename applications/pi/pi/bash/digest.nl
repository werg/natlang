---
description: Shorten a long command output to what matters for the agent's current purpose.
args:
  output: string
  purpose: string
returns: string
model: small
---
output is what a command printed while a coding agent was doing purpose. Keep what the agent needs to act: errors
and warnings with their file:line, failing tests with their assertion messages and expected/actual values, the
summary lines (counts, totals, exit status), and lines that answer purpose. Drop progress bars, repeated lines,
passing tests and boilerplate. Quote kept lines exactly. Stay under 40 lines, and end with one line saying what you
dropped.
