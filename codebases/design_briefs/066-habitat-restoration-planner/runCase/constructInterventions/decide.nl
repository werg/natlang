---
args: { prepared: PreparedEvidence, goal: GoalContract, policy: PolicySelection }
returns: DecisionAtom[]
---
Apply this domain algorithm to prepared evidence: Propose staged interventions from approved guidance and known site conditions. Explain the hypothesized mechanism, dependencies, resource needs, and stop conditions; route unfamiliar or potentially harmful interventions to qualified specialists instead of improvising.
For each field, return a value only when supported, source IDs, authority, one rejected alternative, and status established, disputed, unknown, or inapplicable. Preserve contrary evidence and distinguish observation from inference.
