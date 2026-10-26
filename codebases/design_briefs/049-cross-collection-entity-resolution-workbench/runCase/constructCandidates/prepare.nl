---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for constructCandidates. Generate candidate pairs using exact normalized keys, aliases, and bounded semantic retrieval. Preserve blocking provenance and include audit samples outside candidate blocks so recall failures are measurable rather than hidden by the candidate generator. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
