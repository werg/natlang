---
args: { prepared: PreparedSourcingAgreement, goal: GoalContract, policy: PolicySelection }
returns: DecisionsSourcingAgreement
---
1. Apply this domain algorithm to each relevant evidence unit: Interpret responses as acceptance, rejection, or qualified counteroffer at the term level. Rebuild package feasibility after each material change, identify unresolved conditions, and prepare a final agreement only when all linked commitments are mutually consistent.
2. Return field-level atoms for the actual SourcingAgreement record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
