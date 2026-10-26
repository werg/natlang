---
args: { draft: DraftObservationInterpretation, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated ObservationInterpretation field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Concept assignments have discriminating evidence or remain provisional. Explain the decisive evidence and smallest repair target; never weaken the gate.
