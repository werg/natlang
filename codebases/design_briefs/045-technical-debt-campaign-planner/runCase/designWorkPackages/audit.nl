---
args: { draft: DraftDebtWorkPackage, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated DebtWorkPackage field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Each package is executable and tied to the original symptom or validated mechanism. Explain the decisive evidence and smallest repair target; never weaken the gate.
