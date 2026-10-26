---
args: { assessment: AssessmentBuildRequest }
returns: CompetencyTrace[]
---
1. Bind weightedSkills = inspectCompetencyWeights(assessment).
2. Bind evidencePaths = assignObservableEvidenceTasks(weightedSkills, assessment).
3. Bind nuisanceLoads = separateIncidentalDemands(evidencePaths, assessment), including language, tools, time, and permitted assistance.
4. Preserve competency coverage separately from delivery timing. An accommodation may change time or presentation without changing the target evidence.
