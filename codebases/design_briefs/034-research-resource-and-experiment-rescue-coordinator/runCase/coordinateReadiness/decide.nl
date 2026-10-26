---
args: { prepared: PreparedReadinessCheck, goal: GoalContract, policy: PolicySelection }
returns: DecisionsReadinessCheck
---
1. Apply this domain algorithm to each relevant evidence unit: Collect readiness confirmations from resource owners and investigators, verify current status immediately before execution, and issue role-specific instructions. Preserve cancellations and rescheduling history so downstream analyses know which actual conditions applied.
2. Return field-level atoms for the actual ReadinessCheck record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
