---
args: { draft: DraftDebtMechanism, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated DebtMechanism field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: A popular explanation is not treated as established causality. Explain the decisive evidence and smallest repair target; never weaken the gate.
