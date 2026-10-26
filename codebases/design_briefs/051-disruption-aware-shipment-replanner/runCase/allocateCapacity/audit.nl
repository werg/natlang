---
args: { draft: DraftShippingAllocation, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated ShippingAllocation field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Capacity conservation, route feasibility, and exact cost limits pass. Explain the decisive evidence and smallest repair target; never weaken the gate.
