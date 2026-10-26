---
args: { impact: ObservedIncidentImpact, hypotheses: IncidentCauseSet, diagnostics: BoundedDiagnosticPlan, incident: IncidentRecoveryRequest }
returns: ProposedRecoverySequence
---
Select steps only from the named runbook version. Order prerequisite checks before changes and reversible steps before irreversible ones where the runbook permits. Bind a fresh current-state check and the exact commander approval to every proposed step. If a destructive or improvised command is absent from the authorized runbook, stop and request a reviewed runbook extension.
Bind authority = verifyCommanderApprovalAtProposalTime(incident.commanderId, incident.authorityStatus, incident.incidentEvidence). A stale approval or changed commander blocks the step.
