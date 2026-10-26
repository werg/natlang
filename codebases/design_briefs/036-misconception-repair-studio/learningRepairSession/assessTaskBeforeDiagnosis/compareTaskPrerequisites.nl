---
args: { input: LearnerSessionInput, answers: TaskQualityReview[] }
returns: TaskQualityReview[]
---
Check whether every required prerequisite is included in the supplied competency and learner history. An assessment that depends on an unlearned prerequisite is not valid evidence about the target concept.
