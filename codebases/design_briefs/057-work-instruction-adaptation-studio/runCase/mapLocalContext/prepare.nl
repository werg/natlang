---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: ProcedureRequirement[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for mapLocalContext. Compare local equipment, materials, staffing, and terminology with the master procedure. Separate harmless naming differences from capability changes and identify where local conditions make a step impossible or require engineering review. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
