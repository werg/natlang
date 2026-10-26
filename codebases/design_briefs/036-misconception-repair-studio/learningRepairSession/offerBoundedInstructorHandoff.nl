---
args: { observations: LearnerEvidence[], hypotheses: HypothesisSet[], checkpoint: LearnerCheckpoint, input: LearnerSessionInput }
returns: BoundedHandoff
---
Offer one instructor handoff when the task remains ambiguous, evidence cannot distinguish hypotheses, the supplied retry bound is reached, or the learner signals frustration. Include the last completed checkpoint and unresolved evidence only; do not expose hidden diagnostic labels to the learner.
