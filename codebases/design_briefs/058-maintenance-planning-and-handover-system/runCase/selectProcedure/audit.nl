---
args: { draft: DraftMaintenanceProcedureFit, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated MaintenanceProcedureFit field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Correct asset and procedure edition with all critical prerequisites established. Explain the decisive evidence and smallest repair target; never weaken the gate.
