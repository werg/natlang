---
args: { shiftCase: MaintenanceShiftCase }
returns: ShiftHandoverProposal
---
1. Bind identity = resolveOfficialAsset(shiftCase.assetIdentity). If identity.officialAssetId is null, block dispatch until the asset owner resolves the alias; do not update the asset record from an informal room name.
2. Bind tasks = orderByProcedurePrerequisites(shiftCase.plannedTasks, shiftCase.procedures). Refuse a task whose prerequisite chain is missing or whose asset does not equal the resolved official ID.
3. Bind completion = verifyReportedCompletion(tasks, shiftCase.completionReports). Reported done without required verification remains incomplete, even when the shift has ended.
4. Bind parts = assessRevisionSubstitutions(shiftCase.partChanges, shiftCase.procedures). Keep an unsupported revision substitution out of installation tasks until compatibility evidence is accepted.
5. Bind isolation = shiftCase.isolation. If shiftCase.isolation exists, replace isolation with handOverLiveIsolation(shiftCase.isolation, shiftCase.outgoingShift, shiftCase.incomingShift). Carry exact lock references, state, required handover checks, and named verification ownership; never imply de-isolation.
6. Bind recurrence = carryRecurringFaultWatch(shiftCase.faultHistory, shiftCase.reportedObservations, shiftCase.priorObservation). Compare the preserved prior stage record with later observations; a temporary disappearance leaves the historical fault open under its approved monitoring rule.
7. Return { officialAssetId: identity.officialAssetId, identityBlockers: identity.blockers, orderedTasks: tasks, incompleteTaskIds: completion, rejectedPartChanges: parts, isolationState: isolation, monitoringCarryover: recurrence, handoverNote: "Outgoing " + shiftCase.outgoingShift + " to incoming " + shiftCase.incomingShift, dispatchAllowed: identity.officialAssetId != null && parts.length == 0 && completion.length == 0 }.
