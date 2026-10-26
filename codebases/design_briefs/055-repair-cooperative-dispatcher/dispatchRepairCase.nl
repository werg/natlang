---
args: { repair: CooperativeRepairRequest }
returns: RepairDispatchProposal
---
1. Bind symptom = separateObservedSymptomsFromCause(repair, repair.priorSymptom). Preserve the exact typed prior symptom record and distinguish its observations from newly reported causes.
2. If repair.priorState exists and its symptomVersion matches repair.requestId, resume only the pending monitoring or uncompleted tasks; keep its completedTaskIds and do not repeat completed work.
3. Bind diagnostic = selectApprovedBoundedCheck(symptom, repair). A hazard or absent procedure routes to specialist intake; a workshop check proceeds only when its exact skill and stop conditions are supplied.
4. If diagnostic.route equals "specialist" or diagnostic.route equals "blocked", return a blocked proposal with empty assignments and work steps, a referral/hold reason, and no workshop action.
5. Bind assignments = verifyRepairerFitAndConsent(diagnostic, repair). Expired credentials disqualify only the task requiring them; trainees need named supervision and every assigned task requires consent.
6. If assignments is empty, return a blocked proposal for coordinator review; do not reserve tools or parts.
7. Bind work = scheduleAuthorizedRepairSequence(assignments, repair). Reserve shared tools, reject incompatible parts, and stop for renewed customer approval when cost or scope changes.
8. Bind outcome = verifyRepairAgainstOriginalSymptom(work, symptom, repair). A temporary mask stays open with a recurrence observation, not a verified fix.
9. Return { requestId: repair.requestId, symptom, diagnostic, assignments, work, outcome, checkpoint: { stage: "captureOutcomes", completedTaskIds: work.steps, symptomVersion: repair.requestId, pendingMonitoring: outcome.recurrenceWatch }, commitAllowed: false }.
