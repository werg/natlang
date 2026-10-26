---
args: { observations: LearnerEvidence[], taskQuality: TaskQualityReview[], input: LearnerSessionInput }
returns: HypothesisSet[]
---
Generate only explanations compatible with the observed work. Keep conceptual, arithmetic, reading, incomplete-work, and task-ambiguity alternatives separate. Require at least two independent response patterns before treating a conceptual explanation as more than tentative.
