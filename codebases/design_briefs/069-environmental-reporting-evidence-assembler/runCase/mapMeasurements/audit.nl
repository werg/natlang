---
args: { draft: DraftEnvironmentalMeasure, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated EnvironmentalMeasure field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Every value has method, period, unit, and provenance. Explain the decisive evidence and smallest repair target; never weaken the gate.
