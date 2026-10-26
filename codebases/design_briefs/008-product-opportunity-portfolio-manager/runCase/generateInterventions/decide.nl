---
args: { prepared: PreparedIntervention, goal: GoalContract, policy: PolicySelection }
returns: DecisionsIntervention
---
1. Apply this domain algorithm to each relevant evidence unit: Propose materially different ways to improve each outcome, including process, service, and product changes. Explain the mechanism, affected users, and expected tradeoffs; avoid generating cosmetic variants that all depend on the same untested assumption.
2. Return field-level atoms for the actual Intervention record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
