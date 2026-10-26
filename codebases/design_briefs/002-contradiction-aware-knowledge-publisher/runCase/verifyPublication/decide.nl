---
args: { prepared: PreparedPublicationCheck, goal: GoalContract, policy: PolicySelection }
returns: DecisionsPublicationCheck
---
1. Apply this domain algorithm to each relevant evidence unit: Re-extract claims from the proposed page set and repeat comparison around changed topics. Inspect whether simplification removed exceptions; build a review packet with residual disagreements and publish only after required owners approve the same patch set.
2. Return field-level atoms for the actual PublicationCheck record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
