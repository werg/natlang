---
args: { prepared: PreparedEvidence, goal: GoalContract, policy: PolicySelection }
returns: DecisionAtom[]
---
Apply this domain algorithm to prepared evidence: Combine alerts, user reports, and service telemetry to describe observed impact and uncertainty. Separate symptoms from causes and distinguish missing telemetry from healthy behavior; identify the service outcomes that recovery must actually restore.
For each field, return a value only when supported, source IDs, authority, one rejected alternative, and status established, disputed, unknown, or inapplicable. Preserve contrary evidence and distinguish observation from inference.
