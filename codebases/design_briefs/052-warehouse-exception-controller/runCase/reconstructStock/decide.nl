---
args: { prepared: PreparedEvidence, goal: GoalContract, policy: PolicySelection }
returns: DecisionAtom[]
---
Apply this domain algorithm to prepared evidence: Reconcile manifests, scans, picking notes, and receiving observations while preserving who observed what and when. Separate an item’s physical presence from its expected location and retain uncertainty when labels or serials are missing.
For each field, return a value only when supported, source IDs, authority, one rejected alternative, and status established, disputed, unknown, or inapplicable. Preserve contrary evidence and distinguish observation from inference.
