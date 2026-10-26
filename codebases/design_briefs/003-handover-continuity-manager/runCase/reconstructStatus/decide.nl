---
args: { prepared: PreparedWorkStatus, goal: GoalContract, policy: PolicySelection }
returns: DecisionsWorkStatus
---
1. Apply this domain algorithm to each relevant evidence unit: Match outputs and acknowledgments to each promised outcome. A sent file is not accepted delivery when acceptance was required; cancellation must come from an authorized party, and incomplete evidence produces a status question.
2. Return field-level atoms for the actual WorkStatus record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
