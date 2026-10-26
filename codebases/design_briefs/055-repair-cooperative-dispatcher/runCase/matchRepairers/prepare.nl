---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: RepairDiagnostic[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for matchRepairers. Match tasks to volunteers or specialists using stated skills, verified qualifications where required, availability, and consent. Do not infer capability from identity or enthusiasm, and preserve supervision requirements for trainees. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
