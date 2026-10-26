---
args: { hypotheses: IncidentCauseSet, incident: IncidentRecoveryRequest }
returns: BoundedDiagnosticPlan
---
For each proposed diagnostic require an exact operation ID from the supplied allowlist, a bounded service/query scope, a rate budget, and a hypothesis it distinguishes. Reject unapproved shell text and any query whose estimated load exceeds the declared budget. Intrusive diagnostics remain pending commander approval.
Bind budget = enforceDeclaredQueryBudget(incident.allowlistedOperations, incident.policyClauses). If no approved operation or resource limit is supplied, return no executable diagnostic and ask the commander to authorize a bounded check.
