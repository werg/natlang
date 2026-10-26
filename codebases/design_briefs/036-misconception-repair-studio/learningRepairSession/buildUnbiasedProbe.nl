---
args: { hypotheses: HypothesisSet[], taskQuality: TaskQualityReview[], input: LearnerSessionInput }
returns: UnbiasedDiagnostic[]
---
1. Bind variation = varyOneConceptualFeature(hypotheses, taskQuality).
2. Bind answerSpace = independentlyVerifyAnswerClasses(variation, input).
3. Bind access = rejectUnlearnedPrerequisiteLeak(variation, answerSpace, input).
4. Construct a short diagnostic with the feature varied, unrelated difficulty held constant, hiddenLabels, and evidence that the answer space was independently checked. If the probe cannot distinguish the hypotheses, return a question instead of a diagnostic task.
