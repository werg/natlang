---
args: { draft: DraftDefectObservation, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated DefectObservation field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Every defect observation remains tied to its original inspection context. Explain the decisive evidence and smallest repair target; never weaken the gate.
