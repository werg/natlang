---
args: { draft: DraftObservationLink, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated ObservationLink field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Event counts distinguish duplicate submissions from independent corroboration. Explain the decisive evidence and smallest repair target; never weaken the gate.
