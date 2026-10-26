---
args: { prepared: PreparedResolutionDelta, goal: GoalContract, policy: PolicySelection }
returns: DecisionsResolutionDelta
---
1. Apply this domain algorithm to each relevant evidence unit: Reconcile owner responses against the pending question and current evidence version. Distinguish approval of wording from approval of the decision; append the adopted resolution and invalidate only consumers of the replaced version.
2. Return field-level atoms for the actual ResolutionDelta record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
