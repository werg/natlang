---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: MaintenanceWorkOrder[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for interpretProgress. Compare technician notes and system events with task completion criteria. Distinguish work attempted, work completed, and verification passed; preserve changed conditions and unexpected findings that alter the remaining plan. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
