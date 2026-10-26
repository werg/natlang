---
args: { prepared: PreparedChangeImpact, goal: GoalContract, policy: PolicySelection }
returns: DecisionsChangeImpact
---
1. Apply this domain algorithm to each relevant evidence unit: Compare old and new assertions under their dates, scope, and authority. Determine which premise is actually invalidated and retain the part of the decision whose conditions still hold; never treat a synonym edit as a substantive reversal.
2. Return field-level atoms for the actual ChangeImpact record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
