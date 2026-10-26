---
args: { planning: RestorationPlanningInput }
returns: HabitatActionPlan
---
1. Bind disclosure = restrictSensitiveSpeciesLocations(planning.speciesLocations). Keep exact coordinates only in the authorized conservation view; volunteer and public products use the supplied access rule and coarsened location.
2. Bind comparison = compareSurveySeasons(planning.surveys, planning.baseline). A survey from a different season or method is incomparable until its effect on the indicator is assessed; do not interpret seasonal turnover as restoration change.
3. Bind monitoring = designEffortDenominators(planning.objectives, planning.surveys, planning.volunteerSamples, comparison). Record method, season, duration, and observer count with every volunteer sample so effort changes are not mistaken for abundance changes.
4. Bind permissions = qualifyInterventionAccess(planning.interventions, planning.landPermissions). An expired or mismatched permission stops only field tasks in its area and action scope.
5. Bind impacts = respondToUnexpectedRestorationEffect(planning.impactReports, planning.interventions, planning.expertRoster). Suspend the affected intervention and request qualified review before continuing.
6. Return { caseId: planning.caseId, disclosure, comparison, monitoring, permittedActions: permissions.permittedActions, stoppedActions: permissions.stoppedActions, suspendedInterventions: impacts.suspendedInterventions, expertEscalations: impacts.reviewRequests, unresolvedPermissions: permissions.unresolvedPermissionIds, commitAllowed: false }.
