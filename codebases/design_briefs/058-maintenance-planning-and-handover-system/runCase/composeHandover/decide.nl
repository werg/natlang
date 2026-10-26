---
args: { prepared: PreparedEvidence, goal: GoalContract, policy: PolicySelection }
returns: DecisionAtom[]
---
Apply this domain algorithm to prepared evidence: Prepare a shift handover describing actual asset state, work in progress, unresolved findings, and required next checks. Require acknowledgment from the receiving owner and preserve the distinction between maintenance completion and authorized return to service.
For each field, return a value only when supported, source IDs, authority, one rejected alternative, and status established, disputed, unknown, or inapplicable. Preserve contrary evidence and distinguish observation from inference.
