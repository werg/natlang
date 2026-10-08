---
readout: decision
args:
  instructions: string
  condition: string
  examples: string
returns: ConditionKind
---
A compiler wants to replace part of a natural-language function with code. condition is a test on the function's inputs that, in the recorded examples, picks out the calls handled one particular way. instructions are the function's instructions.

Answer "structural" when a reader following the instructions would decide these calls by the condition itself: a keyword or command the instructions name, a format, a field value, an empty or missing input, a number range.

Answer "semantic" when the instructions ask about meaning and the condition only agrees with that meaning in these examples: a paraphrase, a negation, a synonym or another language would break it.
