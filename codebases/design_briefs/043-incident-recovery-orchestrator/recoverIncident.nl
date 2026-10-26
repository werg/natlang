---
args: { incident: IncidentRecoveryRequest }
returns: IncidentRecoveryProposal
---
1. Bind impact = scopeObservedImpact(incident). Separate user-visible failures from alert state and label unobserved service cohorts unknown.
2. Bind hypotheses = compareCausalExplanations(impact, incident). Keep multiple mechanisms live until a discriminating observation favors one.
3. Bind diagnostics = constrainDiagnosticQueries(hypotheses, incident). Select only named allowlisted operations whose query scope and resource budget are bounded.
4. Bind recovery = sequenceAuthorizedRunbook(impact, hypotheses, diagnostics, incident). Recheck current state and commander authority immediately before each proposed step; a command result alone is not restoration.
5. Bind timedOut = reconcileAmbiguousActionCompletion(incident.pendingActions, incident). Resolve actual action state before offering a retry. If commander changed, transfer authority and approvals explicitly.
6. Bind restoration = verifyUserOutcomeRecovery(recovery, incident). Test representative user functions and cohorts for the supplied observation window; keep data inconsistency or recurrence as open incident work.
7. Return { incidentId: incident.incidentId, impact, hypotheses, diagnostics, recovery, timedOut, restoration, open: restoration.status !== 'restored', commitAllowed: false }.
