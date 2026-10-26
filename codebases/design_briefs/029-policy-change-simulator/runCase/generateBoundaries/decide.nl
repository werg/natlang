---
args: { prepared: PreparedBoundaryScenario, goal: GoalContract, policy: PolicySelection }
returns: DecisionsBoundaryScenario
---
1. Apply this domain algorithm to each relevant evidence unit: Construct near-neighbor cases around changed thresholds, exceptions, and overlapping categories. Vary one consequential fact at a time, include adverse interactions, and identify scenarios whose outcome exposes an ambiguity requiring a policy decision.
2. Return field-level atoms for the actual BoundaryScenario record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
