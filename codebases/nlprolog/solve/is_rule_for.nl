---
description: Is this statement a rule whose conclusion, applied to the goal's subject, would give the goal?
args:
  statement: Text
  goal: Text
returns: Bool
---
`args/statement` is one sentence of a knowledge base; `args/goal` is a claim about something specific.
Answer true if the statement is a general rule (if/then, every, all, whoever, anything that ...) and its conclusion,
applied to the thing the goal talks about, is the goal. Facts are not rules: answer false for them.
