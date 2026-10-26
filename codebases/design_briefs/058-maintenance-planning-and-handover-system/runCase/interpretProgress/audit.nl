---
args: { draft: DraftMaintenanceProgress, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated MaintenanceProgress field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: No completion based solely on free-text optimism or elapsed time. Explain the decisive evidence and smallest repair target; never weaken the gate.
