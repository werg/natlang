---
args: { prepared: PreparedEvidence, goal: GoalContract, policy: PolicySelection }
returns: DecisionAtom[]
---
Apply this domain algorithm to prepared evidence: Construct conditional field and entity mappings that preserve domain meaning. Identify lossy transformations, required target extensions, and records needing manual resolution; do not invent missing facts to satisfy a stricter target schema.
For each field, return a value only when supported, source IDs, authority, one rejected alternative, and status established, disputed, unknown, or inapplicable. Preserve contrary evidence and distinguish observation from inference.
