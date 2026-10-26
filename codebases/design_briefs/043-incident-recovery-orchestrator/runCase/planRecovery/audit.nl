---
args: { draft: DraftRecoveryStep, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated RecoveryStep field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Current authority, service state, and runbook preconditions checked immediately before execution. Explain the decisive evidence and smallest repair target; never weaken the gate.
