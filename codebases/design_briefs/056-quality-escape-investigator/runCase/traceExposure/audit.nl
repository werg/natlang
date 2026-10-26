---
args: { draft: DraftQualityExposure, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated QualityExposure field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Traceability is complete or the uncertainty is explicitly included in containment scope. Explain the decisive evidence and smallest repair target; never weaken the gate.
