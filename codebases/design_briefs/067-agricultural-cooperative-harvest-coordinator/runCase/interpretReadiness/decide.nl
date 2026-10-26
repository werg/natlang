---
args: { prepared: PreparedEvidence, goal: GoalContract, policy: PolicySelection }
returns: DecisionAtom[]
---
Apply this domain algorithm to prepared evidence: Read grower notes and approved assessment records to identify likely readiness, quality, and uncertainty. Distinguish tentative expectations from confirmed availability and preserve variety or lot differences that affect harvest timing and customer suitability.
For each field, return a value only when supported, source IDs, authority, one rejected alternative, and status established, disputed, unknown, or inapplicable. Preserve contrary evidence and distinguish observation from inference.
