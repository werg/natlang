---
args: { draft: DraftObservationQuestion, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated ObservationQuestion field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Questions are informative, feasible, and do not request unsafe field activity. Explain the decisive evidence and smallest repair target; never weaken the gate.
