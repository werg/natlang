---
args: { serviceCase: FacilityServiceCase }
returns: FacilityDispatchPacket
---
1. Bind location = resolveOfficialFacilityLocation(serviceCase.initialObservation, serviceCase.locationAliases, serviceCase.officialLocations). Preserve an informal room name as an alias; never replace the official location identifier.
2. Bind linked = groupByRecurringMechanism(location, serviceCase.initialObservation, serviceCase.repairHistory, serviceCase.relatedReports). Multiple reports may describe one issue; do not reveal reporter identities to other occupants.
3. Bind visit = negotiateConsentBoundVisit(location, serviceCase.accessPermissions, serviceCase.timePatterns). If access fails, record the failed attempt and request a newly consented window rather than silently retrying.
4. Bind work = formMechanismAwareWorkOrder(linked, visit, serviceCase.repairHistory). Schedule observation during the conditions when the complaint occurs; recurring failure escalates to mechanism investigation rather than a disconnected ticket.
5. Bind updates = composePrivacyLimitedOccupantUpdates(linked, serviceCase.updatePreferences). Send only fields each recipient consented to receive.
6. Return { caseId: serviceCase.caseId, officialLocationId: location.locationId, locationBlockers: location.blockers, linkedIssueIds: linked.issueIds, workOrderTasks: work.tasks, visitWindow: visit.windows, accessBlockers: visit.blockers, observationPlan: work.observationPlan, privateUpdatePackets: updates, mechanismEscalation: work.escalation, dispatchAllowed: location.locationId != null && visit.blockers.length == 0, commitAllowed: false }.
