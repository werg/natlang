---
args: { prepared: PreparedAmendmentRelease, goal: GoalContract, policy: PolicySelection }
returns: DecisionsAmendmentRelease
---
1. Apply this domain algorithm to each relevant evidence unit: Present a before-and-after obligation diff and affected-case replay to qualified reviewers. Freeze the approved interpretation, rebase active cases at controlled checkpoints, and preserve old versions for disputes, audit, and rollback of uncommitted changes.
2. Return field-level atoms for the actual AmendmentRelease record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
