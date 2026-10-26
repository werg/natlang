---
args: { output: ApprovedAnalysisOutput, design: EffectiveDesignTimeline[] }
returns: { assignmentMatches: boolean; observationMatches: boolean; populationMatches: boolean; relevantTimelineRefs: string[]; issues: string[] }
---
Compare output assignment and observation units to each effective plan version covering its outcome window. Keep the units distinct: cluster assignment does not make a person-level observation independent. Match population definitions and retain any changed cohort boundary. Return the actual covering plan references and unresolved version or unit conflicts.
