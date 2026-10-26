---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: MonitoringTask[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for adaptManagement. Interpret new observations against baseline and monitoring rules, including alternative explanations such as seasonality or survey changes. Propose bounded plan revisions, preserve unaffected work, and obtain expert approval before changing interventions. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
