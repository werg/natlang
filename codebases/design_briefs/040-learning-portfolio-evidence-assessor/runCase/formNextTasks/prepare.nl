---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: CompetencyFinding[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for formNextTasks. Design the smallest authentic task that fills a consequential evidence gap. Tailor context and resources to the learner while preserving the competency, and specify what would count as success before the new work is submitted. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
