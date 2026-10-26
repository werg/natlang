---
args: { prepared: PreparedClaimConflict, goal: GoalContract, policy: PolicySelection }
returns: DecisionsClaimConflict
---
1. Apply this domain algorithm to each relevant evidence unit: Generate candidates through shared entities and facets, then compare meanings under aligned scope and time. Classify coexistence as legitimate when claims describe different editions or populations; record the minimum incompatible propositions when a real conflict exists.
2. Return field-level atoms for the actual ClaimConflict record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
