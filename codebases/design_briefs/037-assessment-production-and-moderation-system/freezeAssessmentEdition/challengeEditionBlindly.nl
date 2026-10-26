---
args: { items: AssessmentItemDraft[], rubrics: RubricSpec[], competencies: CompetencyTrace[], assessment: AssessmentBuildRequest }
returns: ChallengeAdjudication[]
---
For each item bind independent = solveWithoutKey(item, competencies, assessment), then bind borderlines = replayBorderlineResponses(item, itsRubric, independent). Compare the independent conclusions with the proposed key and rubric. If a correct independent route is absent, classify the defect as a scoring defect; if reasonable readings change the answer, classify it as an item defect. Repair the prompt and scoring rule together, then retain both original and repaired versions. An unresolved correctness defect blocks the edition.
