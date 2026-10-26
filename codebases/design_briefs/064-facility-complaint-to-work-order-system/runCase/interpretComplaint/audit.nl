---
args: { draft: DraftFacilityObservation, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated FacilityObservation field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Location and symptom are established or a targeted clarification is requested. Explain the decisive evidence and smallest repair target; never weaken the gate.
