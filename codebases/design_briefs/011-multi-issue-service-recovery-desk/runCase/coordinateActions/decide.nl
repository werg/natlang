---
args: { prepared: PreparedRecoveryAction, goal: GoalContract, policy: PolicySelection }
returns: DecisionsRecoveryAction
---
1. Apply this domain algorithm to each relevant evidence unit: Sequence approved refunds, replacements, pickups, and communications. Reserve scarce replacements before promising dates, wait for required customer choices, and reconcile uncertain external outcomes rather than retrying irreversible operations blindly.
2. Return field-level atoms for the actual RecoveryAction record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
