---
args: { prepared: PreparedMethodStep, goal: GoalContract, policy: PolicySelection }
returns: DecisionsMethodStep
---
1. Apply this domain algorithm to each relevant evidence unit: Reconstruct the reported procedure from methods, supplements, code descriptions, and notes. Separate explicitly specified parameters from inferred defaults and distinguish the authors’ actual method from a convenient modern replacement.
2. Return field-level atoms for the actual MethodStep record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
