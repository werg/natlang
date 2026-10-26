---
args: { prepared: PreparedEvidence, goal: GoalContract, policy: PolicySelection }
returns: DecisionAtom[]
---
Apply this domain algorithm to prepared evidence: Turn accepted plans into bounded seasonal tasks, shared-tool reservations, and reminders. Adapt to changed availability or observations, preserve agreements across seasons, and record actual completion without treating missed tasks as automatic grounds for exclusion.
For each field, return a value only when supported, source IDs, authority, one rejected alternative, and status established, disputed, unknown, or inapplicable. Preserve contrary evidence and distinguish observation from inference.
