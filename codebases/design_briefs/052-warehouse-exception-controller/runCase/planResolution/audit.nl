---
args: { draft: DraftWarehouseWorkOrder, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated WarehouseWorkOrder field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: No inventory adjustment merely to make a reconciliation balance. Explain the decisive evidence and smallest repair target; never weaken the gate.
