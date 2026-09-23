---
description: Can the goal be derived from a knowledge base of prose facts and
  rules? Forward chaining to a fixed point.
args:
  goal: string
  kb: string[]
returns: Answer
---
function solve(goal, kb) -> Answer

  rule_flags = for each s in kb: is_rule(s)
  fact_flags = for each f in rule_flags: not f                 # exact: use code
  rules = select_by_flags(kb, rule_flags)
  facts = select_by_flags(kb, fact_flags)

  start = { known: facts, derived: [], grew: true }
  final = repeat at most 6 times, until settled(state):        # nothing new was derived in the last round
      state = derive(state, rules)

  hits = for each k in final.known: same_claim(k, goal)
  if any_true(hits): verdict = "yes"
  else: verdict = "unknown"                                    # open world: not derivable is not the same as false
  return { verdict, derived: final.derived }
