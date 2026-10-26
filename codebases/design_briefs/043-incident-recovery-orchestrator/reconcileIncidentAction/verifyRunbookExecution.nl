---
args: { request: IncidentRecoveryRequest, plan: IncidentRecoveryProposal, receipt: IncidentActionReceipt }
returns: IncidentActionAssessment
---
1. Match receipt.actionId and operationId to a pending incident action and a proposed recovery step. The step's runbookVersion, operation allowlist, and commander approval reference must match exactly.
2. Require request.authorityStatus "authorized", the current commanderId to equal receipt.commanderId, and freshStateCheckRefs for the exact preconditions immediately before the action. A former commander or earlier approval cannot authorize this execution.
3. An "executed" receipt counts as complete only when resultEvidenceRefs are present and the operation's postcondition is verified. A command response or process exit alone is not restoration.
4. A "not-executed" receipt may be retried only when the service confirms it did not run, authority and the allowlisted step remain current, and fresh checks are renewed. Unknown or partial outcomes cannot be retried as though nothing happened.
5. Return the exact authority, step, postcondition, and retry blockers.
