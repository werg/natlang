---
args: { draft: DraftCuratedObservation, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated CuratedObservation field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Rare but supported observations are retained; sensitive locations are appropriately generalized. Explain the decisive evidence and smallest repair target; never weaken the gate.
