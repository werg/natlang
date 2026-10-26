---
args: { draft: DraftMaintenanceHandover, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated MaintenanceHandover field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Handover includes all open obligations and current safety-relevant state. Explain the decisive evidence and smallest repair target; never weaken the gate.
