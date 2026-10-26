---
args: { assessment: AssessmentBuildRequest }
returns: AssessmentEditionPlan
---
1. Bind competencies = traceBlueprintEvidence(assessment). Each weighted skill must point to observable evidence, while incidental reading, tools, and assistance are captured as separate constraints.
2. Bind draftedItems = authorAssessmentItems(competencies, assessment). For each item retain expected knowledge, required context, permitted resources, likely alternative readings, and exposure status.
3. Bind answerability = inspectAnswerability(draftedItems, assessment). Keep unsupported knowledge and materially ambiguous readings visible as defects.
4. Bind rubrics = scoreWithAlternativeMethods(draftedItems, competencies, assessment). Define observable partial-credit boundaries and route unfamiliar-but-valid reasoning to semantic review.
5. Bind challenges = challengeEditionBlindly(draftedItems, rubrics, competencies, assessment). Independently solve each item, test reasonable readings and borderline responses, and repair item plus rubric together.
6. Bind repaired = recheckAfterRepairs(challenges, draftedItems, rubrics, competencies, assessment). Re-run weighted competency coverage, answerability, accessibility, duration, and rubric consistency after every material repair.
7. Bind exposureRetirements = retireLeakedItems(assessment). A leaked item is unavailable for new administrations but all historical response and scoring evidence stays preserved.
8. Bind regrade = planConsistentRegrade(rubrics, challenges, assessment). Before a rubric change, identify every response under the old version and propose one consistent correction set; require moderation authorization before applying it.
9. Bind releaseStatus = decideEditionRelease(draftedItems, answerability, rubrics, challenges, repaired, assessment). A model-answer error or unresolved correctness/ambiguity defect blocks release; it can also trigger a controlled retrospective regrade.
10. Return { caseId: assessment.assessmentId, competencies, items: draftedItems, rubrics, challenges, exposureRetirements, regrade, releaseStatus: releaseStatus.status, releaseDecision: releaseStatus, historicalScoringRetained: true, questions: editionQuestions(challenges, regrade), commitAllowed: false }. Timing policy remains distinct from content-equivalence accommodations.
