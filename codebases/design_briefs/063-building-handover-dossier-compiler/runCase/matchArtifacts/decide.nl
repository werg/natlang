---
args: { prepared: PreparedEvidence, goal: GoalContract, policy: PolicySelection }
returns: DecisionAtom[]
---
Apply this domain algorithm to prepared evidence: Compare submitted manuals, inspection records, certificates, and training logs with the actual requirement. Distinguish a blank form from completed evidence and a generic manual from the installed configuration; retain missing signatures or scope mismatches.
For each field, return a value only when supported, source IDs, authority, one rejected alternative, and status established, disputed, unknown, or inapplicable. Preserve contrary evidence and distinguish observation from inference.
