---
args: { prepared: PreparedContributionTask, goal: GoalContract, policy: PolicySelection }
returns: DecisionsContributionTask
---
1. Apply this domain algorithm to each relevant evidence unit: Group missing technical, financial, and implementation contributions by responsible team. Ask precise questions tied to requirements, specify acceptable evidence, and expose dependencies that could make the submission infeasible before the deadline.
2. Return field-level atoms for the actual ContributionTask record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
