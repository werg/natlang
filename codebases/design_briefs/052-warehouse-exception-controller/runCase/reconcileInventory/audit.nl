---
args: { draft: DraftInventoryResolution, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated InventoryResolution field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Physical movement and ledger changes reconcile without duplicate stock creation. Explain the decisive evidence and smallest repair target; never weaken the gate.
