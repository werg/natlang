---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: DiagnosticTask[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for planRepair. Select a suitable explanation and contrastive examples, then create practice with gradually reduced scaffolding. Adapt to observed responses without simply repeating the original answer, and escalate persistent ambiguity to the instructor. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
