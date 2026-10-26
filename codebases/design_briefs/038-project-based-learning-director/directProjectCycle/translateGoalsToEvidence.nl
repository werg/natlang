---
args: { project: ProjectDirectorRequest, seedGoal: ProjectLearningGoal }
returns: LearningEvidenceMap
---
For each required learning goal, state a demonstration a learner can complete with the project's stated scope and available resources. Split compound goals when a single artifact cannot show every skill. Preserve exclusions and prerequisites as boundaries. Mark a goal untestable when no feasible demonstration fits; do not claim that project polish proves individual mastery.
For each compound requirement bind facets = decomposeGoalIntoObservableFacets(goal). Bind evidence = locateLearnerDemonstration(facets, project.sourceEvidence, project.resourceInventory). Keep only evidence paths that directly show the named facet; return absent facets as unsupported goals.
