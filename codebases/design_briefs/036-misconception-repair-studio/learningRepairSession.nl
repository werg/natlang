---
args: { session: LearnerSessionInput, checkpoint: LearnerCheckpoint | null }
returns: LearningRepairSession
---
1. Bind checkpoint = resumeLearningCheckpoint(session.caseId, checkpoint). Restore only completed task IDs, prior hint exposure, open questions, and authorized accommodations. Keep private hypothesis identifiers separate from learner-visible state.
2. Bind taskQuality = assessTaskBeforeDiagnosis(session). If wording, answer space, accessibility, or prerequisite demand is ambiguous, return the repair branch now: ask the instructor to fix the task and produce no learner misconception label.
3. On a clear task, bind observations = interpretLearnerEvidence(session, checkpoint). Compare the response with the independently accepted solution space, including unconventional valid strategies. Record behavior before inferring a learner state.
4. Bind hypotheses = testCompetingExplanations(observations, taskQuality, session). Include accidental error, reading interpretation, incomplete work, task defect, and conceptual alternatives; one wrong answer alone remains tentative.
5. Only when the rival hypotheses differ on a controllable concept, bind diagnostics = buildUnbiasedProbe(hypotheses, taskQuality, session). Vary one conceptual feature, hold unrelated difficulty fixed, independently check every answer class, and hide which hypothesis each response favors.
6. Bind accessibleDiagnostics = preserveCompetencyAcrossAccommodation(diagnostics, session). Change presentation and access support while retaining the exact competency contract and accepted answer space.
7. Bind interventions = planContrastiveRepair(hypotheses, accessibleDiagnostics, checkpoint, session). Use verified explanations, contrastive examples, and gradually reduced scaffolds. Cap repeated attempts and preserve hint exposure for later mastery assessment.
8. Bind transfer = assessNovelUnassistedTransfer(interventions, checkpoint, session). Do not claim mastery from a repeated item, a hinted response, or an accommodated presentation that changed the competency.
9. Bind instructorHandoff = offerBoundedInstructorHandoff(observations, hypotheses, checkpoint, session). Offer one scoped handoff after persistent difficulty or unresolved task ambiguity; never keep the learner in an endless retry loop.
10. Bind nextCheckpoint = saveLearningProgress(checkpoint, observations, hypotheses, interventions, transfer, instructorHandoff). Return { caseId: session.caseId, taskQuality, observations, hypotheses, diagnostics: accessibleDiagnostics, interventions, transfer, checkpoint: nextCheckpoint, instructorHandoff, hiddenLabelsExposed: false, masteryClaim: transfer.some(result => result.demonstratedSkill !== "") }.
