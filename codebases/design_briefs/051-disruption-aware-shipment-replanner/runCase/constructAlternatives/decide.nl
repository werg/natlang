---
args: { prepared: PreparedEvidence, goal: GoalContract, policy: PolicySelection }
returns: DecisionAtom[]
---
Apply this domain algorithm to prepared evidence: Generate permitted rerouting, split-shipment, consolidation, or substitution options. Explain which commitments each option preserves or changes, and exclude alternatives violating handling requirements or relying on unverified carrier capacity.
For each field, return a value only when supported, source IDs, authority, one rejected alternative, and status established, disputed, unknown, or inapplicable. Preserve contrary evidence and distinguish observation from inference.
