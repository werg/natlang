---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: RepairSymptom[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for planDiagnostics. Choose a bounded diagnostic task from supplied safe procedures that distinguishes plausible issues. Specify required qualifications, tools, and stop conditions; route work beyond cooperative capability to an appropriate specialist rather than improvising. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
