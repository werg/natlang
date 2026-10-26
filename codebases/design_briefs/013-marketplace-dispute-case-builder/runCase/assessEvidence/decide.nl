---
args: { prepared: PreparedEvidenceFinding, goal: GoalContract, policy: PolicySelection }
returns: DecisionsEvidenceFinding
---
1. Apply this domain algorithm to each relevant evidence unit: Evaluate relevance, consistency, and provenance of each submitted item against a specific question. Distinguish evidence authenticity checks from semantic support, and discount multiple copies of the same record without dismissing independent corroboration.
2. Return field-level atoms for the actual EvidenceFinding record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
