---
args: { prepared: PreparedContainmentPlan, goal: GoalContract, policy: PolicySelection }
returns: DecisionsContainmentPlan
---
1. Apply this domain algorithm to each relevant evidence unit: Select approved containment actions proportionate to the established issue and uncertainty. Coordinate holds, inspections, alternative supply, and reviewed communications while preserving unaffected orders and avoiding promises before capacity is reserved.
2. Return field-level atoms for the actual ContainmentPlan record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
