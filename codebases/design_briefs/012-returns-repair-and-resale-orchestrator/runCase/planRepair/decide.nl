---
args: { prepared: PreparedRepairPlan, goal: GoalContract, policy: PolicySelection }
returns: DecisionsRepairPlan
---
1. Apply this domain algorithm to each relevant evidence unit: Translate the selected path into authorized inspection or repair tasks using supplied procedures. Check part compatibility and workshop capacity, preserve stop conditions, and create alternative plans when repair would exceed delegated cost or capability.
2. Return field-level atoms for the actual RepairPlan record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
