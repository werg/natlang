---
description: Rewrite the instructions of a legacy leaf judgment in the new style.
args:
  instructions: string
  steps: KindedStep[]
  signature: string
  previous?: string
  problems: string[]
returns: string
---
instructions are the body of a function with signature that answers with one of its result's values. steps are the
same instructions split into steps, each with what it does. Write the new instructions:

1. Keep the meaning of every "judgment" step: what to weigh, against what evidence, and when each result value
   applies. Name the arguments by their names in signature.
2. Leave out "guard-for-type" and "redundant" steps; say a "mechanism" step in one short sentence if the judgment
   needs it.
3. When there are several judgment steps, number them; otherwise write one paragraph.
4. Write only the instructions: no front matter, no code fence, no heading.

If previous is provided, it is your earlier rewrite and problems are what an exact check found wrong with it; fix every
problem while keeping what already worked. Return the new instructions as text.
