---
args: { interventions: LearningIntervention[], checkpoint: LearnerCheckpoint, input: LearnerSessionInput }
returns: TransferFinding[]
---
1. Bind novel = verifyNovelTaskSurface(interventions, input). Exclude rehearsed prompts and altered-number copies.
2. Bind independent = verifyNoHintContamination(novel, checkpoint, input). Separate unassisted work from performance after a hint, worked example, or scaffold.
3. Call updateCompetencyConservatively(independent, input). Claim transfer only for novel evidence meeting the supplied competency standard and retain remaining uncertainty for instructor review.
