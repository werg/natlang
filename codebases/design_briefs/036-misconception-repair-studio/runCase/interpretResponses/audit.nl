---
args: { draft: DraftResponseObservation, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated ResponseObservation field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Observed behavior is separated from inferred learner state. Explain the decisive evidence and smallest repair target; never weaken the gate.
