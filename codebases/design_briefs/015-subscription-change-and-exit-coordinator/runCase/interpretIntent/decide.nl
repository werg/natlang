---
args: { prepared: PreparedChangeIntent, goal: GoalContract, policy: PolicySelection }
returns: DecisionsChangeIntent
---
1. Apply this domain algorithm to each relevant evidence unit: Read the request in conversational context and identify the intended account scope. Distinguish cancelling an add-on from closing the account, and ask a focused question only when ambiguity would change access, billing, or data retention.
2. Return field-level atoms for the actual ChangeIntent record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
