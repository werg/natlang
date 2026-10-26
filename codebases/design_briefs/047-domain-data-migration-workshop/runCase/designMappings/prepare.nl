---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: SourceFieldMeaning[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for designMappings. Construct conditional field and entity mappings that preserve domain meaning. Identify lossy transformations, required target extensions, and records needing manual resolution; do not invent missing facts to satisfy a stricter target schema. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
