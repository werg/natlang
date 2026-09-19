---
description: Apply one rule to the goal - work out its conditions for this case and try to derive each of them.
args:
  rule: Text
  goal: Text
  kb: Text[]
returns: Answer
recursive: true
max_depth: 4
uses:
  solve: ../solve
  all_yes: ./try_rule/all_yes
---
function try_rule(rule, goal, kb) -> Answer

  conditions = conditions_for(rule, goal)       # the rule's conditions, rewritten for the thing the goal is about
  results = for each c in conditions: solve(c, kb)
  return all_yes(results, rule)                  # yes (proof: the rule, then the proofs of the conditions) only if every result is yes
