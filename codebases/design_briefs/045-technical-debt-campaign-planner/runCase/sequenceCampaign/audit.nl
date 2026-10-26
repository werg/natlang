---
args: { draft: DraftDebtWave, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated DebtWave field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Feasible capacity, acyclic dependencies, and safe migration sequencing. Explain the decisive evidence and smallest repair target; never weaken the gate.
