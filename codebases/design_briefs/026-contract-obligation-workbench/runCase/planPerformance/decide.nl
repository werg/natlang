---
args: { prepared: PreparedObligationTask, goal: GoalContract, policy: PolicySelection }
returns: DecisionsObligationTask
---
1. Apply this domain algorithm to each relevant evidence unit: Turn active obligations into bounded tasks with ownership, dependencies, and exact computed dates. Preserve legal interpretation separately from operational scheduling and identify impossible or conflicting task requirements before issuing commitments.
2. Return field-level atoms for the actual ObligationTask record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
