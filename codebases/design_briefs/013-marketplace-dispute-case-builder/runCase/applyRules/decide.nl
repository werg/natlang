---
args: { prepared: PreparedDisputeRecommendation, goal: GoalContract, policy: PolicySelection }
returns: DecisionsDisputeRecommendation
---
1. Apply this domain algorithm to each relevant evidence unit: Apply supplied marketplace rules to established facts and explicitly conditional alternatives for unresolved facts. Explain how the outcome changes under plausible resolutions and route cases outside delegated interpretation to authorized reviewers.
2. Return field-level atoms for the actual DisputeRecommendation record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
