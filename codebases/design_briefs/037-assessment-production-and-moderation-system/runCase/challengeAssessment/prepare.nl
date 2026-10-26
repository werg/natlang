---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: ScoringRubric[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for challengeAssessment. Independently solve each item, attempt reasonable alternate readings, and test rubric consistency on borderline responses. Repair items and rubrics together, then rerun coverage and duration checks before releasing a frozen assessment edition. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
