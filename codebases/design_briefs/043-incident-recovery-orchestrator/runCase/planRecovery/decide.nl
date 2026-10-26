---
args: { prepared: PreparedEvidence, goal: GoalContract, policy: PolicySelection }
returns: DecisionAtom[]
---
Apply this domain algorithm to prepared evidence: Select a recovery path from approved runbooks using current evidence and legal action options. Sequence reversible actions first where appropriate, preserve prerequisites, and avoid improvising destructive commands or treating a successful command as recovered service.
For each field, return a value only when supported, source IDs, authority, one rejected alternative, and status established, disputed, unknown, or inapplicable. Preserve contrary evidence and distinguish observation from inference.
