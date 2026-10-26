---
args: { prepared: PreparedAllocationProposal, goal: GoalContract, policy: PolicySelection }
returns: DecisionsAllocationProposal
---
1. Apply this domain algorithm to each relevant evidence unit: Construct a partial resolution for portions supported by delivery and authorization evidence. Preserve disputed balances, prevent the same receipt from supporting multiple payments, and request targeted confirmation for unresolved acceptance or scope.
2. Return field-level atoms for the actual AllocationProposal record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
