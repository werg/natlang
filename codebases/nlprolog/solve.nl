---
description: Can the goal be derived from the knowledge base? Backward chaining over prose facts and rules.
args:
  goal: Text
  kb: Text[]
returns: Answer
types:
  Answer: '{ verdict: "yes" | "unknown", proof: Text[] }'
recursive: true
max_depth: 4
uses:
  any_true: ../std/any_true
  first_selected: ../std/first_selected
  select_by_flags: ../std/select_by_flags
  first_yes: ./solve/first_yes
---
function solve(goal, kb) -> Answer

  # 1. a fact that says so directly
  stated = for each s in kb: states(s, goal)
  if any_true(stated):
      return { verdict: "yes", proof: [ first_selected(kb, stated) ] }

  # 2. rules whose conclusion would give the goal
  concludes = for each s in kb: is_rule_for(s, goal)
  rules = select_by_flags(kb, concludes)
  if rules is empty:
      return { verdict: "unknown", proof: [] }          # open world: not derivable is not the same as false

  # 3. try every such rule; each one needs all of its conditions
  attempts = for each r in rules: try_rule(r, goal, kb)
  return first_yes(attempts)                              # the first attempt whose verdict is "yes", else unknown
