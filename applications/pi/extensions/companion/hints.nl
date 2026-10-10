---
description: What the agent's reasoning or reply, as it streams in, says it is about to need from the workspace (the companion's stream hint policy, plans/STREAMING.md §2).
args:
  written: string
  known: string[]
returns: StreamHints
---
written is a piece of what a coding agent is writing right now: its reasoning or its reply, still streaming. The
companion reads the files and finds the definitions it returns in the background, so they are ready when the agent
needs them. Each one costs work, so return only what the agent is clearly about to need.

- files: workspace files written names or plainly refers to and is about to read, change or depend on, as paths
  relative to the workspace, at most 3. Include a file that would answer a question the agent asks itself (for
  example "where is the parser configured?" when written or known names the configuration file). known lists files
  the companion already knows; prefer their exact paths.
- symbols: functions, classes, types or other definitions written names and the agent will look for, by their bare
  name (no call parentheses, no module prefix), at most 3.

Return empty lists when written names nothing the agent is about to need: a plan in general words, a summary of
finished work, a reply to the user.
