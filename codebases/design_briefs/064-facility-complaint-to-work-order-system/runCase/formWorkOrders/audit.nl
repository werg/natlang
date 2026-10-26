---
args: { draft: DraftFacilityWorkOrder, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated FacilityWorkOrder field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Task scope and qualification match the observed issue and approved procedure. Explain the decisive evidence and smallest repair target; never weaken the gate.
