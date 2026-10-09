---
description: Which lines of a long tool output the agent should see, and what the whole output says (the companion's output shaping policy).
args:
  call: string
  output: string
  budget: number
returns: OutputShape
---
output is the result of the agent's tool call call (the tool's name and its arguments), too long to show whole: the
agent sees only the lines you keep, at most budget characters together, and can recall the rest. Inspect output with
code (split it into lines, search it) rather than reading all of it.

Keep the lines the agent needs to decide its next step: for test, build or lint output the failures and errors with
their messages, locations and stack frames, and the final summary; for a search the matches; for a listing or a file
the part the call asked about. Drop repetition, progress lines and passing items. Number lines from 1 as
output.split("\n") does; keep ranges as { from, to }, inclusive, in order.

gist: what the output says as a whole, including what you left out (for example "212 tests, 3 failing in
parser.test.ts; the rest pass"), in one or two sentences.
