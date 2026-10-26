---
args: { variation: UnbiasedDiagnostic[], answers: UnbiasedDiagnostic[], input: LearnerSessionInput }
returns: UnbiasedDiagnostic[]
---
Check the diagnostic against the learner's supplied taught-skill record. If the item requires an unlearned prerequisite or creates a new language barrier, withhold it and return a task-revision question.
