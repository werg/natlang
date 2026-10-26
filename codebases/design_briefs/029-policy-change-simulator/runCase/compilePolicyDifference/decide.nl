---
args: { prepared: PreparedPolicyDelta, goal: GoalContract, policy: PolicySelection }
returns: DecisionsPolicyDelta
---
1. Apply this domain algorithm to each relevant evidence unit: Compare baseline and proposed policies at the level of conditions, exceptions, authority, and consequences. Separate editorial changes from changed behavior and identify ambiguous provisions that prevent a determinate comparison.
2. Return field-level atoms for the actual PolicyDelta record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
