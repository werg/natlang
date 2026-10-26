---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: RestorationOption[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for planMonitoring. Design monitoring that can distinguish progress, nonresponse, and unintended effects using supplied methods. Specify comparable locations, timing, and effort, and define what evidence would justify continuing, revising, or stopping an intervention. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
