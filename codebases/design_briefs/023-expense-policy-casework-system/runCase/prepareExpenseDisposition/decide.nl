---
args: { prepared: PreparedExpenseDisposition, goal: GoalContract, policy: PolicySelection }
returns: DecisionsExpenseDisposition
---
1. Apply this domain algorithm to each relevant evidence unit: Produce line-level recommendations and a clear explanation of reductions or holds. Support independent review and appeals with new evidence, then reconcile the approved amount to the reimbursement service without changing the historical original claim.
2. Return field-level atoms for the actual ExpenseDisposition record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
