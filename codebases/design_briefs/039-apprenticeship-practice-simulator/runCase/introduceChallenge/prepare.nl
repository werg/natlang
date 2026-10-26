---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: PracticeTransition[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for introduceChallenge. Select a complication appropriate to the learner’s progress and learning objective. Ensure it changes a meaningful decision, preserves a safe recovery path, and does not retroactively invalidate an action that was correct under earlier visible facts. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
