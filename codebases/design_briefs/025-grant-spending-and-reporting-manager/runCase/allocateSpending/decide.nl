---
args: { prepared: PreparedGrantAllocation, goal: GoalContract, policy: PolicySelection }
returns: DecisionsGrantAllocation
---
1. Apply this domain algorithm to each relevant evidence unit: Match expenditures to eligible activities using descriptions, service periods, and supporting records. Split shared costs only under documented allocation rules, detect duplicate funding attribution, and retain unsupported portions for review.
2. Return field-level atoms for the actual GrantAllocation record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
