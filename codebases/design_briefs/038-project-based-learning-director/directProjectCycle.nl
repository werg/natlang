---
args: { project: ProjectDirectorRequest }
returns: ProjectDirectorProposal
---
1. Bind goals = translateGoalsToEvidence(project, project.seedGoal). Preserve the supplied source learning-goal record as the baseline; a broad aspiration has no project status until a learner can demonstrate it in an observable artifact or individual performance.
2. Bind concept = designCollaborativeChallenge(goals, project). Keep learner roles interdependent but separately attributable; when a material disappears, compare replacement routes by the goal evidence they preserve.
3. Bind calendar = sequenceMilestonesAgainstCapacity(concept, project, project.prior). Bind dependencyOrder = orderMilestoneDependencies(calendar.milestones); add dependencyOrder to calendar and mark infeasible milestones when their dependencies or resource limits cannot fit the remaining calendar.
4. Bind scaffolds = adaptSupportWithoutAuthorship(calendar, project, project.prior). Keep generated examples and decompositions outside the learner's assessable evidence. If group authorship is unclear, request a separately observed demonstration for each learner.
5. Bind complication = rehearseLateComplication(scaffolds, calendar, project). Recompute only the remaining feasible work; retain mandatory goal coverage or block release. A learner's creative route is compared against the competency evidence, not a sample artifact.
6. Bind checkpoint = saveProjectCheckpoint(project.prior, calendar, scaffolds). Bind readiness = chooseProjectStatus(goals, calendar, complication). Return { projectId: project.projectId, goals, concept, calendar, scaffolds, complication, checkpoint, readiness, commitAllowed: false }.
