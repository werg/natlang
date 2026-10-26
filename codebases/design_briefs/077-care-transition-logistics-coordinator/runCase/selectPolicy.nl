---
description: selectPolicy for Care Transition Logistics Coordinator.
args: { goal: GoalContract; candidates: EvidenceItem[] }
returns: PolicySelection
---
Select applicable policy clauses using stated authority and effective dates. Preserve conflicts and missing precedence instead of resolving them by recency or document order.
