---
args: { prepared: PreparedRescueOption, goal: GoalContract, policy: PolicySelection }
returns: DecisionsRescueOption
---
1. Apply this domain algorithm to each relevant evidence unit: When a resource or sample becomes unavailable, generate alternatives such as resequencing, qualified substitution, or postponement. Explicitly state scientific deviations and seek investigator approval where the original protocol would no longer hold.
2. Return field-level atoms for the actual RescueOption record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
