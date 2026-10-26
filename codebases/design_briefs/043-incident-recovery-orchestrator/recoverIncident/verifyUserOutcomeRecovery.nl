---
args: { sequence: ProposedRecoverySequence, incident: IncidentRecoveryRequest }
returns: UserOutcomeRestoration
---
Check each affected user-visible function across representative cohorts using independent outcome evidence. Keep infrastructure health indicators as supporting signals only. If data remains inconsistent, recurrence appears, or the supplied observation window is incomplete, keep the incident open and create a separate repair or monitoring obligation.
Bind missingCoverage = identifyUnverifiedUserCohorts(sequence, incident.priorImpact, incident.observationWindow). A green infrastructure metric cannot close any unverified cohort.
