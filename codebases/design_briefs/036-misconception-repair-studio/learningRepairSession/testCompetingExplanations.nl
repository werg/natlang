---
args: { observations: LearnerEvidence[], taskQuality: TaskQualityReview[], input: LearnerSessionInput }
returns: HypothesisSet[]
---
1. Bind candidates = enumerateRivalHypotheses(observations, taskQuality, input), including task ambiguity and accidental error whenever evidence permits.
2. Bind probe = chooseDiscriminatingQuestion(candidates, observations).
3. Keep candidates tentative until a response to that question distinguishes them. If taskQuality is unclear, return no learner diagnosis and direct the next step to task repair.
