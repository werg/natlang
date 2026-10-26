---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: ShippingCommitment[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for constructAlternatives. Generate permitted rerouting, split-shipment, consolidation, or substitution options. Explain which commitments each option preserves or changes, and exclude alternatives violating handling requirements or relying on unverified carrier capacity. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
