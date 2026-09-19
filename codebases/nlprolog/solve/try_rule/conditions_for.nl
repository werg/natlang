---
description: The conditions of a rule, each written out as a claim about the goal's subject.
args:
  rule: Text
  goal: Text
returns: Text[]
---
`args/rule` is a general rule; `args/goal` is the specific claim it would establish.
List what would have to be true for the rule to give the goal: one short, self-contained sentence per condition,
with the rule's variable replaced by the specific thing the goal is about.
Example: rule "Every bird that is not injured can fly.", goal "Tweety can fly." gives
["Tweety is a bird.", "Tweety is not injured."]
