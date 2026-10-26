---
args: { prepared: PreparedEvidence, goal: GoalContract, policy: PolicySelection }
returns: DecisionAtom[]
---
Apply this domain algorithm to prepared evidence: Compare required functions with bid inclusions and exclusions, recognizing different descriptions of the same work. Separate supply, installation, commissioning, and certification responsibilities instead of assuming one broad line covers all phases.
For each field, return a value only when supported, source IDs, authority, one rejected alternative, and status established, disputed, unknown, or inapplicable. Preserve contrary evidence and distinguish observation from inference.
