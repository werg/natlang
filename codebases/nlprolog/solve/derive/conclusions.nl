---
description: The specific facts one rule yields, given what is known.
args:
  rule: Text
  known: Text[]
returns: Text[]
---
`args/rule` is a general rule and `args/known` is a list of facts.
For every specific thing or person for which ALL conditions of the rule are among the known facts, write the rule's
conclusion about it as one short sentence ("Tweety can fly."). Do not list anything that is already known.
If the rule's conditions are not all met for anything, the answer is the empty list.
