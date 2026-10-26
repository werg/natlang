---
args: { prepared: PreparedConditionAssessment, goal: GoalContract, policy: PolicySelection }
returns: DecisionsConditionAssessment
---
1. Apply this domain algorithm to each relevant evidence unit: Reconcile customer descriptions, inspection notes, and supplied images or structured image annotations. Distinguish cosmetic damage, functional symptoms, and unverified allegations; preserve uncertainty and request evidence that changes routing rather than generic photo uploads.
2. Return field-level atoms for the actual ConditionAssessment record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
