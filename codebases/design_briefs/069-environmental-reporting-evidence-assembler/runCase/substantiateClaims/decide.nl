---
args: { prepared: PreparedEvidence, goal: GoalContract, policy: PolicySelection }
returns: DecisionAtom[]
---
Apply this domain algorithm to prepared evidence: Compare proposed narrative claims with actual measured outcomes and activity evidence. Separate completion of an initiative from demonstrated environmental benefit, qualify estimated results, and reject persuasive claims unsupported by the reporting data.
For each field, return a value only when supported, source IDs, authority, one rejected alternative, and status established, disputed, unknown, or inapplicable. Preserve contrary evidence and distinguish observation from inference.
