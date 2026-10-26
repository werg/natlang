---
args: { prepared: PreparedSubscriptionOperation, goal: GoalContract, policy: PolicySelection }
returns: DecisionsSubscriptionOperation
---
1. Apply this domain algorithm to each relevant evidence unit: Verify consent to the exact option and effective date, then apply changes through idempotent adapters. Recheck account versions before commitment and reconcile partial changes across linked products using compensating actions where permitted.
2. Return field-level atoms for the actual SubscriptionOperation record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
