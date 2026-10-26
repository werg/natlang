---
args: { prepared: PreparedCapabilityMatch, goal: GoalContract, policy: PolicySelection }
returns: DecisionsCapabilityMatch
---
1. Apply this domain algorithm to each relevant evidence unit: Compare required outcomes with verified capabilities and their scope. Do not stretch an adjacent feature into compliance; identify exceptions, delivery dependencies, and evidence that would substantiate a currently unverified claim.
2. Return field-level atoms for the actual CapabilityMatch record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
