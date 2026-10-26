---
args: { draft: DraftRepairWorkPlan, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated RepairWorkPlan field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: No unauthorized repair scope, double-booked resource, or unapproved part substitution. Explain the decisive evidence and smallest repair target; never weaken the gate.
