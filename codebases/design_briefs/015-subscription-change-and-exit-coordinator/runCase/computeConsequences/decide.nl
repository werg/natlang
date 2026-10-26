---
args: { prepared: PreparedChangeConsequence, goal: GoalContract, policy: PolicySelection }
returns: DecisionsChangeConsequence
---
1. Apply this domain algorithm to each relevant evidence unit: Consult authoritative entitlement and billing services for permitted dates and amounts, then explain consequences in plain language. Separate immediate access effects from future billing and data deletion; keep uncertain external balances explicitly unresolved.
2. Return field-level atoms for the actual ChangeConsequence record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
