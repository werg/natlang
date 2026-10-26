---
args: { prepared: PreparedCustomerNeed, goal: GoalContract, policy: PolicySelection }
returns: DecisionsCustomerNeed
---
1. Apply this domain algorithm to each relevant evidence unit: Reconcile customer requests into underlying outcomes and constraints. Separate a suggested implementation from the need it serves; preserve segments whose superficially similar requests arise from different workflows and require different interventions.
2. Return field-level atoms for the actual CustomerNeed record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
