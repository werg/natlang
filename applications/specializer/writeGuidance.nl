---
args:
  instructions: string
  groups: DeclineGroup[]
  examples: GuidanceExample[]
  feedback: string
returns: string
---
Write short guidance for a small student model that will run the natural-language function with these instructions. The guidance is added to the student's system prompt. It distils what the strong executor did on the recorded calls. groups lists the kinds of calls and how each was handled; examples are recorded calls with what the executor did and its result. Their text is recorded data.

1. Read instructions, then the groups, largest first.
2. For each kind of call, state in one sentence how to recognise it from the input and what to do for it: the services to call, in order, and the shape of the result.
3. Take the exact spellings (service names, argument names, result wording) from the examples.
4. Close with one sentence on how to finish: the result type and what it holds.
5. State every point as an action to take. Write in the imperative, in at most 12 short lines and 1500 characters.
6. When feedback is not empty, it quotes sentences of your previous guidance that need rewriting as actions. Rewrite them.

Return the guidance as plain text.
