---
args: { prepared: PreparedSubstituteCandidate, goal: GoalContract, policy: PolicySelection }
returns: DecisionsSubstituteCandidate
---
1. Apply this domain algorithm to each relevant evidence unit: Search catalogs and supplier descriptions for equivalent functions, including alternative mechanisms where permitted. Explain how each candidate would serve the use, identify required adaptations, and avoid treating a familiar name or matching dimension as sufficient.
2. Return field-level atoms for the actual SubstituteCandidate record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
