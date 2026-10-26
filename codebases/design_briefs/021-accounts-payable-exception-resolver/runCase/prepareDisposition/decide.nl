---
args: { prepared: PreparedPayableDisposition, goal: GoalContract, policy: PolicySelection }
returns: DecisionsPayableDisposition
---
1. Apply this domain algorithm to each relevant evidence unit: Prepare an auditable recommendation to pay, hold, request correction, or apply a verified credit. Route by entity and delegated thresholds, and explain why undisputed portions may proceed without treating the whole invoice as resolved.
2. Return field-level atoms for the actual PayableDisposition record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
