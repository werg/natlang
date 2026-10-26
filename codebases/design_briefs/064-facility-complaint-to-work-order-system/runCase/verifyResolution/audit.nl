---
args: { draft: DraftFacilityResolution, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated FacilityResolution field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Closure requires the defined outcome evidence or an explicit accepted monitoring state. Explain the decisive evidence and smallest repair target; never weaken the gate.
