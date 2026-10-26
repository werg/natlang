---
args: { prepared: PreparedEvidenceClaim, goal: GoalContract, policy: PolicySelection }
returns: DecisionsEvidenceClaim
---
1. Apply this domain algorithm to each relevant evidence unit: Group evidence by the actual proposition and relevant population rather than broad topic. Identify incompatible measurements and contradictory findings, qualify applicability, and expose evidence gaps without treating absent studies as proof of no effect.
2. Return field-level atoms for the actual EvidenceClaim record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
