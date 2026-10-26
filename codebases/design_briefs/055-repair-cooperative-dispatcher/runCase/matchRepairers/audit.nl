---
args: { draft: DraftRepairAssignment, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated RepairAssignment field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Exact scheduling and required qualification checks pass. Explain the decisive evidence and smallest repair target; never weaken the gate.
