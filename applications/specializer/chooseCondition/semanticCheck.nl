---
readout: decision
args:
  instructions: string
  condition: string
  examples: string
returns: ConditionKind
---
A compiler wants to replace part of a natural-language function with code. condition is a test on the function's inputs that, in the recorded examples, picks out the calls handled one particular way. instructions are the function's instructions.

Answer "structural" when a reader following the instructions would decide these calls by the condition itself: a keyword or command the instructions name, a format, a field value, an empty or missing input, a number range. A condition that fixes the whole form of the input (for example: the entire request is the word refund followed by a number) is structural, because an input of exactly that form has no other plausible reading.

Answer "semantic" when the instructions ask about meaning and the condition only agrees with that meaning in these examples: a word found anywhere inside free text, where a paraphrase, a negation ("do not refund"), a synonym or another language would break it.
