---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: HarvestCommitment[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for matchLots. Compare lot characteristics and readiness with customer requirements, identifying compatible combinations and gaps. Preserve uncertain quality findings and request targeted verification rather than allocating a lot solely because its crop name matches. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
