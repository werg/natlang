---
description: Moderate one post against a written policy - allow, warn, remove, or escalate to a human.
args:
  post: Text
  policy: Text
returns: Decision
types:
  Severity: '"low" | "high"'
  Decision: '{ action: "allow" | "warn" | "remove" | "escalate", rules: Text[], note: Text }'
uses:
  select_by_flags: ../std/select_by_flags
  any_true: ../std/any_true
---
function moderate(post, policy) -> Decision

  rules = split_rules(policy)                                   # exact: one rule per line
  flags = for each r in rules: violates(r, post)
  hits  = select_by_flags(rules, flags)
  if hits is empty:
      return { action: "allow", rules: [], note: "No rule applies." }

  severities = for each r in hits: severity_of(r, post)
  if no severity is "high":                                     # exact: use code
      return { action: "warn", rules: hits, note: "Minor: the author is reminded of the rules." }

  # Removal is drastic: get a second opinion from a stricter reader before acting.
  strict = a copy of violates, with "When in doubt, answer true." replaced by
           "Answer true only if the violation is unmistakable. When in doubt, answer false."
  confirmed_flags = for each r in hits: strict(r, post)
  confirmed = select_by_flags(hits, confirmed_flags)
  if confirmed is empty:
      return { action: "escalate", rules: hits, note: "A serious rule may apply, but it is not clear-cut. A human should look." }
  return { action: "remove", rules: confirmed, note: "A serious violation, confirmed on a strict reading." }
