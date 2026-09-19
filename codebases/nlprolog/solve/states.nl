---
description: Does this one statement, on its own, assert the goal as a fact?
args:
  statement: Text
  goal: Text
returns: Bool
---
`args/statement` is one sentence of a knowledge base; `args/goal` is a claim.
Answer true only if the statement is a plain fact that says the same thing as the goal (wording may differ).
A rule ("if ...", "every ...", "whoever ...") is not a fact: answer false for rules, even when they are about the goal.
