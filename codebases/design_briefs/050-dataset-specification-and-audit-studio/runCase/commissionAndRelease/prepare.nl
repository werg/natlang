---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: CoverageFinding[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for commissionAndRelease. Create targeted acquisition or annotation tasks, rerun audits on new material, and assemble a release manifest with known limits. Freeze splits by source and world family, prevent leakage, and require owner approval for the specific release use. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
