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
9. Treat absent repair.workReceipts as an empty observed receipt list. Bind observedCompletedTaskIds to receipt.taskId only when receipt.symptomVersion equals repair.requestId, receipt.plannedStep exactly matches one authorized work step, receipt.disposition is "completed", receipt.verificationRefs are present, and every observedCheckId names a supplied repair.outcomeChecks entry for the original symptom. A proposed step, assigned repairer, tool reservation, or schedule never completes a task. Preserve priorState.completedTaskIds only when priorState.symptomVersion equals repair.requestId.
10. Bind checkpoint = { stage: "captureOutcomes", completedTaskIds: unique(priorState completed IDs still valid for this symptom plus observedCompletedTaskIds), symptomVersion: repair.requestId, pendingMonitoring: outcome.recurrenceWatch plus each authorized work step without a matching verified receipt }. Return { requestId: repair.requestId, symptom, diagnostic, assignments, work, outcome, checkpoint, commitAllowed: false }.
