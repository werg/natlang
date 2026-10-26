---
args: { draft: DraftMaintenanceWorkOrder, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated MaintenanceWorkOrder field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Exact resource feasibility and no bypass of required safety or qualification checks. Explain the decisive evidence and smallest repair target; never weaken the gate.
