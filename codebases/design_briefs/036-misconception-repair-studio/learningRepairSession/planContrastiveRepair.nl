---
args: { hypotheses: HypothesisSet[], diagnostics: UnbiasedDiagnostic[], checkpoint: LearnerCheckpoint, input: LearnerSessionInput }
returns: LearningIntervention[]
---
1. Bind explanation = selectVerifiedExplanation(hypotheses, input).
2. Bind examples = constructContrastiveExamples(explanation, diagnostics, input); contrast the learner's observed strategy with a valid concept-preserving strategy without repeating the original answer.
3. Bind practice = scheduleFadingPractice(examples, checkpoint, input). Reduce scaffold only after the prior response demonstrates the prerequisite.
4. Stop after the supplied attempt bound. Persist frustration and unresolved difficulty for the instructor instead of restarting the same practice loop.
