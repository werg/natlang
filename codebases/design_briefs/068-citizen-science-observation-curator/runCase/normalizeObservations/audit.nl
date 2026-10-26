---
args: { draft: DraftFieldObservation, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated FieldObservation field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Original evidence and uncertainty remain recoverable. Explain the decisive evidence and smallest repair target; never weaken the gate.
