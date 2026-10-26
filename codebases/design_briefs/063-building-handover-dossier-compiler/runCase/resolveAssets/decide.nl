---
args: { prepared: PreparedEvidence, goal: GoalContract, policy: PolicySelection }
returns: DecisionAtom[]
---
Apply this domain algorithm to prepared evidence: Reconcile asset tags, drawing labels, manuals, and contractor naming conventions. Distinguish identical models at different locations, preserve serial and revision data, and flag ambiguous identity before attaching a certificate or manual to the wrong asset.
For each field, return a value only when supported, source IDs, authority, one rejected alternative, and status established, disputed, unknown, or inapplicable. Preserve contrary evidence and distinguish observation from inference.
