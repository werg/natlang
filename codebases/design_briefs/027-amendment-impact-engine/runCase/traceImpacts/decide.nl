---
args: { prepared: PreparedAmendmentImpact, goal: GoalContract, policy: PolicySelection }
returns: DecisionsAmendmentImpact
---
1. Apply this domain algorithm to each relevant evidence unit: Traverse dependency edges from changed obligation facets, not the whole agreement. Assess which open tasks, approvals, notices, and future schedules become stale while retaining completed actions that were valid under their historical effective rules.
2. Return field-level atoms for the actual AmendmentImpact record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
