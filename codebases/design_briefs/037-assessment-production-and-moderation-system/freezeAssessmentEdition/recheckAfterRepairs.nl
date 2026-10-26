---
args: { challenges: ChallengeAdjudication[], items: AssessmentItemDraft[], rubrics: RubricSpec[], competencies: CompetencyTrace[], assessment: AssessmentBuildRequest }
returns: RepairRegressionReport
---
Recompute the competency-to-item map after each repair. Check that no weighted competency disappeared, no excluded construct became necessary, each remaining item is answerable, and changed response formats still fit the published duration and approved accommodation. Compare the modified rubric against every independent solution and borderline class. Return the exact checks repeated and any newly introduced defect; a repair clears the old issue only when the revised item and its rubric pass together.
