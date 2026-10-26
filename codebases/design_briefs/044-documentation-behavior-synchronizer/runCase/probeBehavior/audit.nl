---
args: { draft: DraftBehaviorObservation, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated BehaviorObservation field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Probe is reproducible and isolates the behavior under review. Explain the decisive evidence and smallest repair target; never weaken the gate.
