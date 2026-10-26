---
args: { prepared: PreparedExpenseLine, goal: GoalContract, policy: PolicySelection }
returns: DecisionsExpenseLine
---
1. Apply this domain algorithm to each relevant evidence unit: Extract receipt lines and claimant explanations while preserving original currency and dates. Distinguish the merchant’s description from the claimed business purpose, and flag illegible or missing evidence rather than reconstructing amounts through guesswork.
2. Return field-level atoms for the actual ExpenseLine record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
